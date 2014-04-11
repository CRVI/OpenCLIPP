////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform_Buffer.cl
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformation on image buffers
//! 
//! Copyright (C) 2014 - CRVI
//!
//! This file is part of OpenCLIPP.
//! 
//! OpenCLIPP is free software: you can redistribute it and/or modify
//! it under the terms of the GNU Lesser General Public License version 3
//! as published by the Free Software Foundation.
//! 
//! OpenCLIPP is distributed in the hope that it will be useful,
//! but WITHOUT ANY WARRANTY; without even the implied warranty of
//! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//! GNU Lesser General Public License for more details.
//! 
//! You should have received a copy of the GNU Lesser General Public License
//! along with OpenCLIPP.  If not, see <http://www.gnu.org/licenses/>.
//! 
////////////////////////////////////////////////////////////////////////////////

#include "Buffers.h"

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#define _UNARY_OP(name, code, type, suffix) \
   kernel void CONCATENATE(name, suffix) (INPUT_SPACE const type * source, global type * dest, int src_step, int dst_step, int width, int height)\
   {\
      BEGIN\
      src_step /= sizeof(type);\
      dst_step /= sizeof(type);\
      code;\
   }

#define UNARY_OP(name, code) \
   _UNARY_OP(name, code, SCALAR, _1C)\
   _UNARY_OP(name, code, TYPE2,  _2C)\
   _UNARY_OP(name, code, TYPE3,  _3C)\
   _UNARY_OP(name, code, TYPE4,  _4C)\

UNARY_OP(mirror_x, dest[gy * dst_step + gx] = source[gy * src_step + width - gx - 1])
UNARY_OP(mirror_y, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + gx])
UNARY_OP(flip, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + width - gx - 1])

//UNARY_OP(transpose, dest[gx * dst_step + gy] = source[gy * src_step + gx])  // This naive transpose version is very slow


kernel void set_all_1C(global SCALAR * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(SCALAR);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_2C(global TYPE2 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE2);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_3C(global TYPE3 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE3);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_4C(global TYPE4 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE4);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}


// Implementation of fast GPU image transposition
// Uses a local temporary buffer to store a block of data
// Then does coalesced writes to dest using the data from the temp buffer
// Inspired by http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/

#define LW 32
#define LH 8
#define WIDTH1 4

#define TRANSPOSE_IMPL(type, suffix) \
   __attribute__((reqd_work_group_size(LW, LH, 1)))\
   kernel void CONCATENATE(transpose, suffix) (INPUT_SPACE const type * source, global type * dest, int src_step, int dst_step, int width, int height)\
   {\
      src_step /= sizeof(type);\
      dst_step /= sizeof(type);\
      local type temp[LW][LW + 1];\
      const int local_x = get_local_id(0);\
      const int local_y = get_local_id(1);\
      const int srcx = get_group_id(0) * LW + local_x;\
      const int srcy = get_group_id(1) * LW + local_y;\
      const int dstx = get_group_id(1) * LW + local_x;\
      const int dsty = get_group_id(0) * LW + local_y;\
      if (srcx < width)\
         for (int j = 0; j < LW; j += LH)\
            if ((srcy + j) < height)\
               temp[local_y + j][local_x] = source[(srcy + j) * src_step + srcx];\
      barrier(CLK_LOCAL_MEM_FENCE);\
      if (dstx >= height || dsty >= width)\
         return;\
      for (int j = 0; j < LW; j += LH)\
         if ((dsty + j) < width)\
            dest[(dsty + j) * dst_step + dstx] = temp[local_x][local_y + j];\
   }

#define TRANSPOSE_FLUSH_IMPL(type, suffix) \
   __attribute__((reqd_work_group_size(LW, LH, 1)))\
   kernel void CONCATENATE(transpose_flush, suffix) (INPUT_SPACE const type * source, global type * dest, int src_step, int dst_step, int width, int height)\
   {\
      src_step /= sizeof(SCALAR);\
      dst_step /= sizeof(SCALAR);\
      local type temp[LW][LW + 1];\
      const int local_x = get_local_id(0);\
      const int local_y = get_local_id(1);\
      const int srcx = get_group_id(0) * LW + local_x;\
      const int srcy = get_group_id(1) * LW + local_y;\
      const int dstx = get_group_id(1) * LW + local_x;\
      const int dsty = get_group_id(0) * LW + local_y;\
      for (int j = 0; j < LW; j += LH)\
         temp[local_y + j][local_x] = source[(srcy + j) * src_step + srcx];\
      barrier(CLK_LOCAL_MEM_FENCE);\
      for (int j = 0; j < LW; j += LH)\
         dest[(dsty + j) * dst_step + dstx] = temp[local_x][local_y + j];\
   }

// Standard version with bounds checking
TRANSPOSE_IMPL(SCALAR, _1C)
TRANSPOSE_IMPL(TYPE2,  _2C)
TRANSPOSE_IMPL(TYPE3,  _3C)
TRANSPOSE_IMPL(TYPE4,  _4C)

// Faster version for images that have a width and height that are multiples of 32
TRANSPOSE_FLUSH_IMPL(SCALAR, _1C)
TRANSPOSE_FLUSH_IMPL(TYPE2,  _2C)
TRANSPOSE_FLUSH_IMPL(TYPE3,  _3C)
TRANSPOSE_FLUSH_IMPL(TYPE4,  _4C)
