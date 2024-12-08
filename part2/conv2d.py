import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output_tile should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

   batch_size, in_channels, input_height, input_width = X.shape
   out_channels, in_channels_, filter_height, filter_width = W.shape
   out_channels_ = bias.shape[0]

   assert (
       in_channels_ == in_channels and out_channels_ == out_channels
   ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

   out_height = input_height - filter_height + 1
   out_width = input_width - filter_width + 1

   out_pool_height = out_height // pool_size
   out_pool_width = out_width // pool_size
   
   # Can assume multiple of 128 to avoid using mask
   assert in_channels % 128 == 0

   # Can assume one PSUM bank can at least fit one row of the pixels
   assert nl.tile_size.gemm_moving_fmax >= out_width

   # Initialize output_tile array
   X_out = nl.ndarray(
       shape=(batch_size, out_channels, out_pool_height, out_pool_width),
       dtype=X.dtype,
       buffer=nl.hbm,
   )

   # Various tiling dimensions (You may want to define more of them)
   c_in_pmax = 128
   c_out_pmax = 128

   n_tiles_c_in = in_channels // 128
   n_tiles_c_out = out_channels // 128 

   
   #preloading form
   w_old = nl.ndarray(
       shape=(n_tiles_c_out, nl.par_dim(c_out_pmax), n_tiles_c_in, 128, filter_height, filter_width),
       dtype=W.dtype,
       buffer=nl.sbuf
   )

   w_new = nl.ndarray(
       shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_out_pmax), c_in_pmax),
       dtype=W.dtype,
       buffer=nl.sbuf
   )

   w_new2 = nl.ndarray(
       shape=(filter_height, filter_width, n_tiles_c_out, n_tiles_c_in, nl.par_dim(c_in_pmax), c_out_pmax),
       dtype=W.dtype,
       buffer=nl.sbuf
   )

   # load weights
   for cout in nl.affine_range(n_tiles_c_out):         
        for cin in nl.affine_range(n_tiles_c_in):
            w_old[cout, :, cin, :, :, :] = nl.load(W[cout * 128: cout * 128 + 128, cin * 128 : cin * 128 + 128, :, :])

   # transpose weights
   for kh in nl.affine_range(filter_height):
        for kw in nl.affine_range(filter_width):
            for cout in nl.affine_range(n_tiles_c_out):
                for cin in nl.affine_range(n_tiles_c_in):
                    w_new[kh, kw, cout, cin] = nl.copy(w_old[cout, :, cin, :, kh, kw])
                    w_new2[kh, kw, cout, cin] = nisa.nc_transpose(w_new[kh, kw, cout, cin])
   

   # Handle output row tiling for large images
   output_tile_height = 2
   input_tile_height = output_tile_height + filter_height - 1
   
   n_tiles_h = out_height // output_tile_height

   for img in nl.affine_range(batch_size):
        for tile_h in nl.affine_range(n_tiles_h):
            # assign space in SBUF to store image row tile
            X_tile = nl.ndarray(shape=(n_tiles_c_in, nl.par_dim(c_in_pmax), input_tile_height, input_width), 
                dtype=X[img].dtype, 
                buffer=nl.sbuf)

            for cin in nl.affine_range(n_tiles_c_in):
                X_tile[cin, :, :, :] = nl.load(X[img, cin * 128: cin * 128 + 128, tile_h * output_tile_height: tile_h * output_tile_height + input_tile_height, :])            
            
            for cout in nl.affine_range(n_tiles_c_out):
                
                # asign space in SBUF to store output_tile
                output_tile = nl.ndarray(shape=(nl.par_dim(c_out_pmax), output_tile_height, out_width), 
                    dtype=X_out[img].dtype, 
                    buffer=nl.sbuf)
                
                for out_row in nl.affine_range(output_tile_height):
                    # assign space in PSUM to store output_tile row
                  temp = nl.zeros((nl.par_dim(c_out_pmax), out_width), nl.float32, buffer=nl.psum)
                
                    for kh in nl.affine_range(filter_height):
                        for kw in nl.affine_range(filter_width):
                            for cin in nl.affine_range(n_tiles_c_in):
                                w_slice = w_new2[kh, kw, cout, cin, :, :]
                                x_slice = X_tile[cin, :, out_row + kh, kw : kw + out_width]
                                
                                # Perform matrix multiplication and accumulate in PSUM
                                temp += nl.matmul(w_slice, x_slice, transpose_x=True)
                    
                    output_tile[:, out_row, :] = temp

                bias_tile = nl.load(bias[cout*128:(cout+1) * 128]).reshape((128, 1))
                output_tile = nisa.tensor_scalar(output_tile, np.add, bias_tile)
                
                if pool_size > 1:                    
                    sz_p, sz_hin, sz_win = output_tile.shape

                    # output_tile 
                    i_0 = nl.arange(sz_p)[:, None, None, None, None] #
                    i_1 = nl.arange(sz_hin//pool_size)[None, :, None, None, None] # y_outer
                    i_2 = nl.arange(pool_size)[None, None, :, None, None] # y_inner
                    i_3 = nl.arange(sz_win//pool_size)[None, None, None, :, None] # x_outer
                    i_4 = nl.arange(pool_size)[None, None, None, None, :] # x_inner

                    out_tile = nl.max(output_tile[i_0, pool_size*i_1+i_2, pool_size*i_3+i_4], axis=[2,4])

                    h_start = tile_h
                    h_end = h_start + 1

                    nl.store(
                        X_out[img, cout * 128: cout * 128 + 128, h_start:h_end , :],
                        value=out_tile,
                    )
                else:
                    h_start = tile_h * output_tile_height
                    h_end = h_start + output_tile_height
                    nl.store(
                        X_out[img, cout * 128: cout * 128 + 128, h_start:h_end , :],
                        value=output_tile,
                    )
   return X_out
