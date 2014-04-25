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



struct SImage
{
   uint Width;    ///< Width of the image, in pixels
   uint Height;   ///< Height of the image, in pixels
   uint Step;     ///< Nb of bytes between each row
   uint Channels; ///< Number of channels in the image, allowed values : 1, 2, 3 or 4

   /// EDataType : Lists possible types of data
   int Type;  ///< Data type of each channel in the image
};


#define SAMPLE_NN(type, suffix) \
   type CONCATENATE(sample_nn, suffix) (INPUT_SPACE const type * source, int src_step, float2 pos, int width, int height)\
   {\
      int2 ipos = convert_int2(pos);\
      return source[ipos.y * src_step + ipos.x];\
   }

#define SAMPLE_LINEAR(type, float_type, suffix) \
   type CONCATENATE(sample_linear, suffix) (INPUT_SPACE const type * source, int src_step, float2 pos, int width, int height)\
   {\
      pos -= (float2)(0.5f, 0.5f);\
      int x1 = (int)(pos.x);\
      float factorx1 = 1 - (pos.x - x1);\
      int x2 = (int)(pos.x + 1);\
      float factorx2 = 1 - factorx1;\
      int y1 = (int)(pos.y);\
      float factory1 = 1 - (pos.y - y1);\
      int y2 = (int)(pos.y + 1);\
      float factory2 = 1 - factory1;\
      float_type f1 = factorx1 * factory1;\
      float_type f2 = factorx2 * factory1;\
      float_type f3 = factorx1 * factory2;\
      float_type f4 = factorx2 * factory2;\
      float_type v1 = CONCATENATE(convert_, float_type)(source[y1 * src_step + x1]);\
      float_type v2 = CONCATENATE(convert_, float_type)(source[y1 * src_step + x2]);\
      float_type v3 = CONCATENATE(convert_, float_type)(source[y2 * src_step + x1]);\
      float_type v4 = CONCATENATE(convert_, float_type)(source[y2 * src_step + x2]);\
      return CONCATENATE(convert_, type)(v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4);\
   }


#define ROTATE(name, sampling, type, suffix) \
   kernel void CONCATENATE(name, suffix) (INPUT_SPACE const type * source, global type * dest,\
      struct SImage src_img, struct SImage dst_img, \
      float sina, float cosa, float xshift, float yshift)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(type);\
      int dst_step = dst_img.Step / sizeof(type);\
      if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)\
         return;\
      float srcx = gx - xshift;\
      float srcy = gy - yshift;\
      float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);\
      type value = 0;\
      if (srcpos.x >= 0.5f && srcpos.x + 0.5f < src_img.Width && srcpos.y >= 0.5f && srcpos.y + 0.5f < src_img.Height)\
         value = CONCATENATE(sampling, suffix) (source, src_step, srcpos, src_img.Width, src_img.Height);\
      dest[pos.y * dst_step + pos.x] = value;\
   }


#define RESIZE(name, sampling, type, suffix) \
   kernel void CONCATENATE(name, suffix)(INPUT_SPACE const type * source, global type * dest,\
      struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(type);\
      int dst_step = dst_img.Step / sizeof(type);\
      if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)\
         return;\
      float2 srcpos = {(pos.x + 0.4995f) * ratioX, (pos.y + 0.4995f) * ratioY};\
      type value = 0;\
      if ((int)(srcpos.x + .5f) == src_img.Width)\
         srcpos.x = src_img.Width - 0.5001f;\
      if ((int)(srcpos.y + .5f) == src_img.Height)\
         srcpos.y = src_img.Height - 0.5001f;\
      if (srcpos.x >= -1 && srcpos.x < src_img.Width - .5f && srcpos.y >= -1 && srcpos.y < src_img.Height - .5f)\
         value = CONCATENATE(sampling, suffix) (source, src_step, srcpos, src_img.Width, src_img.Height);\
      dest[pos.y * dst_step + pos.x] = value;\
   }


SAMPLE_NN(SCALAR, _1C)
SAMPLE_NN(TYPE2,  _2C)
SAMPLE_NN(TYPE3,  _3C)
SAMPLE_NN(TYPE4,  _4C)

SAMPLE_LINEAR(SCALAR, float,  _1C)
SAMPLE_LINEAR(TYPE2,  float2, _2C)
SAMPLE_LINEAR(TYPE3,  float3, _3C)
SAMPLE_LINEAR(TYPE4,  float4, _4C)

ROTATE(rotate_img, sample_nn, SCALAR, _1C)
ROTATE(rotate_img, sample_nn, TYPE2,  _2C)
ROTATE(rotate_img, sample_nn, TYPE3,  _3C)
ROTATE(rotate_img, sample_nn, TYPE4,  _4C)

ROTATE(rotate_linear, sample_linear, SCALAR, _1C)
ROTATE(rotate_linear, sample_linear, TYPE2,  _2C)
ROTATE(rotate_linear, sample_linear, TYPE3,  _3C)
ROTATE(rotate_linear, sample_linear, TYPE4,  _4C)

RESIZE(resize, sample_nn, SCALAR, _1C)
RESIZE(resize, sample_nn, TYPE2 , _2C)
RESIZE(resize, sample_nn, TYPE3 , _3C)
RESIZE(resize, sample_nn, TYPE4 , _4C)

RESIZE(resize_linear, sample_linear, SCALAR, _1C)
RESIZE(resize_linear, sample_linear, TYPE2 , _2C)
RESIZE(resize_linear, sample_linear, TYPE3 , _3C)
RESIZE(resize_linear, sample_linear, TYPE4 , _4C)
