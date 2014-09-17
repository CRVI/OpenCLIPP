////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.cl
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformation
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

#define INPUT  INPUT_SPACE const TYPE *
#define OUTPUT global TYPE *

#define UNARY_OP(name, code) \
   kernel void name (INPUT source, OUTPUT dest, int src_step, int dst_step, int width, int height)\
   {\
      BEGIN\
      src_step /= sizeof(TYPE);\
      dst_step /= sizeof(TYPE);\
      code;\
   }

UNARY_OP(mirror_x, dest[gy * dst_step + gx] = source[gy * src_step + width - gx - 1])
UNARY_OP(mirror_y, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + gx])
UNARY_OP(flip, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + width - gx - 1])

//UNARY_OP(transpose, dest[gx * dst_step + gy] = source[gy * src_step + gx])  // This naive transpose version is very slow


kernel void set_all(OUTPUT dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_rect(OUTPUT dest, int dst_step, uint X, uint Y, uint Width, uint Height, float value)
{
   BEGIN
   if (gx < X || gy < Y || gx >= X + Width || gy >= Y + Height)
      return;

   dst_step /= sizeof(TYPE);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}



// Implementation of fast GPU image transposition
// Uses a local temporary buffer to store a block of data
// Then does coalesced writes to dest using the data from the temp buffer
// Inspired by http://devblogs.nvidia.com/parallelforall/efficient-matrix-transpose-cuda-cc/

#define LW 32
#define LH 8
#define WIDTH1 4

// Standard version with bounds checking
__attribute__((reqd_work_group_size(LW, LH, 1)))
kernel void transpose(INPUT source, OUTPUT dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(TYPE);
   local TYPE temp[LW][LW + 1];
   const int local_x = get_local_id(0);
   const int local_y = get_local_id(1);
   const int srcx = get_group_id(0) * LW + local_x;
   const int srcy = get_group_id(1) * LW + local_y;
   const int dstx = get_group_id(1) * LW + local_x;
   const int dsty = get_group_id(0) * LW + local_y;
   if (srcx < width)
      for (int j = 0; j < LW; j += LH)
         if ((srcy + j) < height)
            temp[local_y + j][local_x] = source[(srcy + j) * src_step + srcx];

   barrier(CLK_LOCAL_MEM_FENCE);

   if (dstx >= height || dsty >= width)
      return;

   for (int j = 0; j < LW; j += LH)
      if ((dsty + j) < width)
         dest[(dsty + j) * dst_step + dstx] = temp[local_x][local_y + j];
}

// Faster version for images that have a width and height that are multiples of 32
__attribute__((reqd_work_group_size(LW, LH, 1)))
kernel void transpose_flush(INPUT source, OUTPUT dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(TYPE);
   local TYPE temp[LW][LW + 1];
   const int local_x = get_local_id(0);
   const int local_y = get_local_id(1);
   const int srcx = get_group_id(0) * LW + local_x;
   const int srcy = get_group_id(1) * LW + local_y;
   const int dstx = get_group_id(1) * LW + local_x;
   const int dsty = get_group_id(0) * LW + local_y;
   for (int j = 0; j < LW; j += LH)
      temp[local_y + j][local_x] = source[(srcy + j) * src_step + srcx];

   barrier(CLK_LOCAL_MEM_FENCE);

   for (int j = 0; j < LW; j += LH)
      dest[(dsty + j) * dst_step + dstx] = temp[local_x][local_y + j];
}



struct SImage
{
   uint Width;    ///< Width of the image, in pixels
   uint Height;   ///< Height of the image, in pixels
   uint Step;     ///< Nb of bytes between each row
   uint Channels; ///< Number of channels in the image, allowed values : 1, 2, 3 or 4

   /// EDataType : Lists possible types of data
   int Type;  ///< Data type of each channel in the image
};



TYPE sample_nn(INPUT source, int src_step, float2 pos, int2 SrcSize)
{
   int2 ipos = convert_int2(pos);
   if (ipos.x < 0 || ipos.x >= SrcSize.x || ipos.y < 0 || ipos.y >= SrcSize.y)
      return 0;

   return source[ipos.y * src_step + ipos.x];
}

TYPE sample_linear(INPUT source, int src_step, float2 pos, int2 SrcSize)
{
   if ((int)(pos.x + .5f) == SrcSize.x)
      pos.x = SrcSize.x - 0.5001f;

   if ((int)(pos.y + .5f) == SrcSize.y)
      pos.y = SrcSize.y - 0.5001f;

   pos -= (float2)(0.5f, 0.5f);

   if (pos.x < -0.5f || pos.x >= SrcSize.x - 1 || pos.y < -0.5f || pos.y >= SrcSize.y - 1)
      return 0;
   
   int x1 = (int)(pos.x);
   float factorx1 = 1 - (pos.x - x1);
   int x2 = (int)(pos.x + 1);
   float factorx2 = 1 - factorx1;
   int y1 = (int)(pos.y);
   float factory1 = 1 - (pos.y - y1);
   int y2 = (int)(pos.y + 1);
   float factory2 = 1 - factory1;
   REAL f1 = factorx1 * factory1;
   REAL f2 = factorx2 * factory1;
   REAL f3 = factorx1 * factory2;
   REAL f4 = factorx2 * factory2;
   REAL v1 = CONVERT_REAL(source[y1 * src_step + x1]);
   REAL v2 = CONVERT_REAL(source[y1 * src_step + x2]);
   REAL v3 = CONVERT_REAL(source[y2 * src_step + x1]);
   REAL v4 = CONVERT_REAL(source[y2 * src_step + x2]);
   return CONVERT(v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4);
}


TYPE sample_bicubic_border(INPUT source, int src_step, float2 pos, int2 SrcSize)
{
   int2 isrcpos = convert_int2(pos);
   float dx = pos.x - isrcpos.x;
   float dy = pos.y - isrcpos.y;

   REAL C[4] = {0, 0, 0, 0};

   if (isrcpos.x < 0 || isrcpos.x >= SrcSize.x)
      return 0;

   if (isrcpos.y < 0 || isrcpos.y >= SrcSize.y)
      return 0;

   for (int i = 0; i < 4; i++)
   {
      int y = isrcpos.y - 1 + i;
      if (y < 0)
         y = 0;

      if (y >= SrcSize.y)
         y = SrcSize.y - 1;

      int Middle = clamp(isrcpos.x, 0, SrcSize.x - 1);

      REAL center = CONVERT_REAL(source[y * src_step + Middle]);

      REAL left = 0, right1 = 0, right2 = 0;
      if (isrcpos.x - 1 >= 0)
         left = CONVERT_REAL(source[y * src_step + isrcpos.x - 1]);
      else
         left = center;

      if (isrcpos.x + 1 < SrcSize.x)
         right1 = CONVERT_REAL(source[y * src_step + isrcpos.x + 1]);
      else
         right1 = center;

      if (isrcpos.x + 2 < SrcSize.x)
         right2 = CONVERT_REAL(source[y * src_step + isrcpos.x + 2]);
      else
         right2 = right1;

      REAL a0 = center;
      REAL d0 = left - a0;
      REAL d2 = right1 - a0;
      REAL d3 = right2 - a0;

      REAL a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
      REAL a2 =  1.0f / 2 * d0 + 1.0f / 2 * d2;
      REAL a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
      C[i] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
   }

   REAL d0 = C[0] - C[1];
   REAL d2 = C[2] - C[1];
   REAL d3 = C[3] - C[1];
   REAL a0 = C[1];
   REAL a1 = -1.0f / 3 * d0 + d2 -1.0f / 6 * d3;
   REAL a2 = 1.0f / 2 * d0 + 1.0f / 2 * d2;
   REAL a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
   return CONVERT(a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy);
}

TYPE sample_bicubic(INPUT source, int src_step, float2 pos, int2 SrcSize)
{
   pos -= (float2)(0.5f, 0.5f);

   int2 isrcpos = convert_int2(pos);
   float dx = pos.x - isrcpos.x;
   float dy = pos.y - isrcpos.y;

   if (isrcpos.x <= 0 || isrcpos.x >= SrcSize.x - 2)
      return sample_bicubic_border(source, src_step, pos, SrcSize);

   if (isrcpos.y <= 0 || isrcpos.y >= SrcSize.y - 2)
      return sample_bicubic_border(source, src_step, pos, SrcSize);

   REAL C[4] = {0, 0, 0, 0};

   for (int i = 0; i < 4; i++)
   {
      const int y = isrcpos.y - 1 + i;
      REAL a0 = CONVERT_REAL(source[y * src_step + isrcpos.x]);
      REAL d0 = CONVERT_REAL(source[y * src_step + isrcpos.x - 1]) - a0;
      REAL d2 = CONVERT_REAL(source[y * src_step + isrcpos.x + 1]) - a0;
      REAL d3 = CONVERT_REAL(source[y * src_step + isrcpos.x + 2]) - a0;

      REAL a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
      REAL a2 =  1.0f / 2 * d0 + 1.0f / 2 * d2;
      REAL a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
      C[i] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
   }

   REAL d0 = C[0] - C[1];
   REAL d2 = C[2] - C[1];
   REAL d3 = C[3] - C[1];
   REAL a0 = C[1];
   REAL a1 = -1.0f / 3 * d0 + d2 -1.0f / 6 * d3;
   REAL a2 = 1.0f / 2 * d0 + 1.0f / 2 * d2;
   REAL a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
   return CONVERT(a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy);
}


#define ROTATE(name, sampling) \
   kernel void name (INPUT source, OUTPUT dest,\
      struct SImage src_img, struct SImage dst_img, \
      float sina, float cosa, float xshift, float yshift)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(TYPE);\
      int dst_step = dst_img.Step / sizeof(TYPE);\
      if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)\
         return;\
      float srcx = gx - xshift;\
      float srcy = gy - yshift;\
      float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);\
      int2 SrcSize = (int2)(src_img.Width, src_img.Height);\
      TYPE value = 0;\
      if (srcpos.x >= 0.5f && srcpos.x + 0.5f < src_img.Width && srcpos.y >= 0.5f && srcpos.y + 0.5f < src_img.Height)\
         value = sampling (source, src_step, srcpos, SrcSize);\
      dest[pos.y * dst_step + pos.x] = value;\
   }


#define RESIZE(name, sampling) \
   kernel void name(INPUT source, OUTPUT dest,\
      struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(TYPE);\
      int dst_step = dst_img.Step / sizeof(TYPE);\
      if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)\
         return;\
      float2 srcpos = {(pos.x + 0.4995f) * ratioX, (pos.y + 0.4995f) * ratioY};\
      int2 SrcSize = (int2)(src_img.Width, src_img.Height);\
      TYPE value = sampling (source, src_step, srcpos, SrcSize);\
      dest[pos.y * dst_step + pos.x] = value;\
   }

#define SHEAR(name, sampling) \
   kernel void name(INPUT source, OUTPUT dest,\
      struct SImage src_img, struct SImage dst_img,\
      float shearx, float sheary, float factor, float xshift, float yshift)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(TYPE);\
      int dst_step = dst_img.Step / sizeof(TYPE);\
      if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)\
         return;\
      float posx = pos.x - xshift;\
      float posy = pos.y - yshift;\
      float2 srcpos = {\
         (posx - posy * shearx) * factor + .5f,\
         (posy - posx * sheary) * factor + .5f};\
      int2 SrcSize = (int2)(src_img.Width, src_img.Height);\
      TYPE value = sampling (source, src_step, srcpos, SrcSize);\
      dest[pos.y * dst_step + pos.x] = value;\
   }

#define REMAP(name, sampling) \
   kernel void name(INPUT source, INPUT_SPACE const float * xmap, INPUT_SPACE const float * ymap,\
      OUTPUT dest, uint xmap_step, uint ymap_step, struct SImage src_img, struct SImage dst_img)\
   {\
      BEGIN\
      int src_step = src_img.Step / sizeof(TYPE);\
      int dst_step = dst_img.Step / sizeof(TYPE);\
      xmap_step /= sizeof(float);\
      ymap_step /= sizeof(float);\
      float2 srcpos = {xmap[gy * xmap_step + gx] + .5f, ymap[gy * ymap_step + gx] + .5f};\
      int2 SrcSize = (int2)(src_img.Width, src_img.Height);\
      TYPE value = sampling (source, src_step, srcpos, SrcSize);\
      dest[pos.y * dst_step + pos.x] = value;\
   }

ROTATE(rotate_nn, sample_nn)
ROTATE(rotate_linear, sample_linear)
ROTATE(rotate_bicubic, sample_bicubic)

RESIZE(resize_nn, sample_nn)
RESIZE(resize_linear, sample_linear)
RESIZE(resize_bicubic, sample_bicubic)

SHEAR(shear_nn, sample_nn)
SHEAR(shear_linear, sample_linear)
SHEAR(shear_cubic, sample_bicubic)

REMAP(remap_nn, sample_nn)
REMAP(remap_linear, sample_linear)
REMAP(remap_cubic, sample_bicubic)


TYPE lanczos(INPUT source, global const float * factors, int src_step, float2 pos, int2 SrcSize, int a, int size)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   pos -= (float2)(0.5f, 0.5f);

   int2 isrcpos = convert_int2(pos);

   REAL sum = 0;
   float weighty = 0;

   float xterms[6] = {0};

   for (int ix = 0; ix < a * 2; ix++)
      xterms[ix] = factors[gx + ix * size];

   global const float * factors_y = factors + size * 2 * a;

   // Add up terms across the filter.
   for (int iy = 0; iy < a * 2; iy++)
   {
      int y = isrcpos.y + iy - a + 1;
      y = clamp(y, 0, SrcSize.y - 1);

      REAL sumx = 0;
      float weightx = 0;

      // Filter along X
      for (int ix = 0; ix < a * 2; ix++)
      {
         int x = isrcpos.x + ix - a + 1;
         x = clamp(x, 0, SrcSize.x - 1);

         float lanc_term = xterms[ix];
         sumx += CONVERT_REAL(source[y * src_step + x]) * lanc_term;
         weightx += lanc_term;
      }

      // Normalize the result
      sumx /= weightx;

      // Sum the results along Y
      float lanc_termy = factors_y[gy + iy * size];
      sum += sumx * lanc_termy;
      weighty += lanc_termy;
   }

   // Normalize the result
   sum /= weighty;

   return CONVERT(sum);
}



#define PI 3.14159265359f

float sinc(float x)
{
   x = (x * PI);
   return sin(x) / x;
}

float Lanczos(float x, int a)
{      
   if (x >= a || x <= -a)
      return 0;

   if (x == 0)
      return 1;
   
   return sinc(x) * sinc(x / a);
}


// These functions pre-calculate the lanczos factors for the rows and column of the image to be resized

void calculate_lanczos_factors(global float * factors, float ratio, int a, int size)
{
   int pos = get_global_id(0);      // Row or column number, 0 to max(Width, Height))
   int index = get_global_id(1);    // pixel index, 0 to 2*a

   float src_pos = (pos + 0.4995f) * ratio - 0.5f;
   int isrc_pos = (int) src_pos;
   int sample = isrc_pos - a + 1 + index;
   factors[pos + index * size] = Lanczos(sample - src_pos, a);
}

void prepare_resize_lanczos(global float * factors, float ratioX, float ratioY, int a, int size)
{
   int dir_x = get_global_id(2);    // x or y, 0 or 1

   if (dir_x)
      calculate_lanczos_factors(factors, ratioX, a, size);
   else
      calculate_lanczos_factors(factors + size * 2 * a, ratioY, a, size);
}

kernel void prepare_resize_lanczos2(global float * factors, float ratioX, float ratioY, int size)
{
   prepare_resize_lanczos(factors, ratioX, ratioY, 2, size);
}

kernel void prepare_resize_lanczos3(global float * factors, float ratioX, float ratioY, int size)
{
   prepare_resize_lanczos(factors, ratioX, ratioY, 3, size);
}


void resize_lanczos(INPUT source, global const float * factors, OUTPUT dest,
   struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY, int a, int size)
{
   BEGIN
   int src_step = src_img.Step / sizeof(TYPE);
   int dst_step = dst_img.Step / sizeof(TYPE);
   if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)
      return;

   float2 srcpos = {(pos.x + 0.4995f) * ratioX, (pos.y + 0.4995f) * ratioY};
   int2 SrcSize = (int2)(src_img.Width, src_img.Height);
   TYPE value = lanczos(source, factors, src_step, srcpos, SrcSize, a, size);
   dest[pos.y * dst_step + pos.x] = value;
}

kernel void resize_lanczos2(INPUT source, global const float * factors, OUTPUT dest,
   struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY, int size)
{
   resize_lanczos(source, factors, dest, src_img, dst_img, ratioX, ratioY, 2, size);
}

kernel void resize_lanczos3(INPUT source, global const float * factors, OUTPUT dest,
   struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY, int size)
{
   resize_lanczos(source, factors, dest, src_img, dst_img, ratioX, ratioY, 3, size);
}


TYPE supersample_border(INPUT source, int src_step, float2 pos, int2 SrcSize, float2 ratio)
{
   REAL sum = 0;
   float factor_sum = 0;

   float2 start = pos - ratio / 2;
   float2 end = start + ratio;
   int2 istart = convert_int2(start);
   int2 length = convert_int2(end - convert_float2(istart)) + 1;

   float2 factors = 1.f / ratio;

   for (int iy = 0; iy < length.y; iy++)
   {
      int y = istart.y + iy;

      if (y < 0 || y >= SrcSize.y)
         continue;

      float factor_y = factors.y;
      if (y < start.y)
         factor_y = factors.y * (y + 1 - start.y);

      if (y + 1 > end.y)
         factor_y = factors.y * (end.y - y);

      for (int ix = 0; ix < length.x; ix++)
      {
         int x = istart.x + ix;

         if (x < 0 || x >= SrcSize.x)
            continue;

         float factor_x = factors.x;
         if (x < start.x)
            factor_x = factors.x * (x + 1 - start.x);

         if (x + 1 > end.x)
            factor_x = factors.x * (end.x - x);

         float factor = factor_x * factor_y;

         sum += CONVERT_REAL(source[y * src_step + x]) * factor;

         factor_sum += factor;
      }

   }

   sum /= factor_sum;
   
   return CONVERT(sum);
}

TYPE supersample(INPUT source, int src_step, float2 pos, int2 SrcSize, float ratioX, float ratioY)
{
   float2 ratio = (float2)(ratioX, ratioY);

   REAL sum = 0;

   float2 start = pos - ratio / 2;
   float2 end = start + ratio;
   int2 istart = convert_int2(start);
   int2 length = convert_int2(end - convert_float2(istart)) + 1;

   float2 factors = 1.f / ratio;

   if (start.x < 0 || start.y < 0 || end.x > SrcSize.x || end.y > SrcSize.y)
      return supersample_border(source, src_step, pos, SrcSize, ratio);

   for (int iy = 0; iy < length.y; iy++)
   {
      int y = istart.y + iy;

      float factor_y = factors.y;
      if (y < start.y)
         factor_y = factors.y * (y + 1 - start.y);

      if (y + 1 > end.y)
         factor_y = factors.y * (end.y - y);

      for (int ix = 0; ix < length.x; ix++)
      {
         int x = istart.x + ix;

         float factor_x = factors.x;
         if (x < start.x)
            factor_x = factors.x * (x + 1 - start.x);

         if (x + 1 > end.x)
            factor_x = factors.x * (end.x - x);

         float factor = factor_x * factor_y;

         sum += CONVERT_REAL(source[y * src_step + x]) * factor;
      }

   }
   
   return CONVERT(sum);
}

kernel void resize_supersample(INPUT source, OUTPUT dest,
   struct SImage src_img, struct SImage dst_img, float ratioX, float ratioY)
{
   BEGIN
   int src_step = src_img.Step / sizeof(TYPE);
   int dst_step = dst_img.Step / sizeof(TYPE);
   if (pos.x >= dst_img.Width || pos.y >= dst_img.Height)
      return;

   float2 srcpos = {(pos.x + 0.4995f) * ratioX, (pos.y + 0.4995f) * ratioY};
   int2 SrcSize = (int2)(src_img.Width, src_img.Height);
   TYPE value = supersample(source, src_step, srcpos, SrcSize, ratioX, ratioY);
   dest[pos.y * dst_step + pos.x] = value;
}
