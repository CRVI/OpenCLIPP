////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityFFT.cl
//! @date   : Feb 2014
//!
//! @brief  : Image comparison for pattern matching, accelerated using FFT
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

#ifndef NBCHAN
#define GET_CHAN(ptr, channel) (*(ptr))
#define SET_CHAN(ptr, channel, value) (*(ptr)) = value
#define SELECT_CHAN(val) val.x
#else
#if NBCHAN == 2
#define GET_CHAN(ptr, channel) (channel == 1 ? (*(ptr)).x : (*(ptr)).y)
#define SET_CHAN(ptr, channel, value) \
   if (channel == 1)\
      (*(ptr)).x = value;\
   else\
      (*(ptr)).y = value;
#define SELECT_CHAN(val) val.xy
#else if NBCHAN == 4
#define GET_CHAN(ptr, channel) (channel == 1 ? (*(ptr)).x : (channel == 2 ? (*(ptr)).y : (channel == 3 ? (*(ptr)).z : (*(ptr)).w)))
#define SET_CHAN(ptr, channel, value) \
   if (channel == 1)\
      (*(ptr)).x = value;\
   else if (channel == 2)\
      (*(ptr)).y = value;\
   else if (channel == 3)\
      (*(ptr)).z = value;\
   else\
      (*(ptr)).w = value;
#define SELECT_CHAN(val) val
#endif
#endif

REAL AreaSqSum(INPUT_SPACE const REAL * SqSumImage, int src_step, int w, int h, int dest_width, int dest_height)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);	

   int Left   = gx     - w / 2 - 1;
   int Right  = gx + w - w / 2 - 1;
   int Top    = gy     - h / 2 - 1;
   int Bottom = gy + h - h / 2 - 1;
   
   Right  = clamp(Right,  0, dest_width  - 1);
   Bottom = clamp(Bottom, 0, dest_height - 1);

   REAL TopLeft = 0;
   REAL TopRight = 0;
   REAL BottomLeft = 0;

   if (Top >= 0 && Left >= 0)
      TopLeft = SqSumImage[Top * src_step + Left];

   if (Top >= 0)
      TopRight = SqSumImage[Top * src_step + Right];
   
   if (Left >= 0)
      BottomLeft = SqSumImage[Bottom * src_step + Left];

   REAL BottomRight = SqSumImage[Bottom * src_step + Right];

   return (BottomRight - TopRight) - (BottomLeft - TopLeft);
}

kernel void square_difference(INPUT_SPACE const REAL * source, global REAL * dest, int src_step, int dst_step,
                              int dest_width, int dest_height, int w, int h, float4 templ_sqsum)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step /= sizeof(REAL);
   dst_step /= sizeof(REAL);

   REAL TemplateSqSum = SELECT_CHAN(templ_sqsum);

   // Compute the square sum of the area
   REAL v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);

   // Pre-computed cross correlation
   REAL ccorr = dest[gy * dst_step + gx];

   // Compute the square difference
   dest[gy * dst_step + gx] = v - 2.f * ccorr + TemplateSqSum;
}

kernel void square_difference_norm(INPUT_SPACE const REAL * source, global REAL * dest, int src_step, int dst_step,
                                   int dest_width, int dest_height, int w, int h, float4 templ_sqsum)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step /= sizeof(REAL);
   dst_step /= sizeof(REAL);

   REAL TemplateSqSum = SELECT_CHAN(templ_sqsum);

   // Compute the square sum of the area
   REAL v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);

   // Pre-computed cross correlation
   REAL ccorr = dest[gy * dst_step + gx];

   // Compute the normalized square difference
   dest[gy * dst_step + gx] = (v - 2.f * ccorr + TemplateSqSum) / sqrt(v * TemplateSqSum);
}

kernel void crosscorr_norm(INPUT_SPACE const REAL * source, global REAL * dest, int src_step, int dst_step,
                           int dest_width, int dest_height, int w, int h, float4 templ_sqsum)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step /= sizeof(REAL);
   dst_step /= sizeof(REAL);

   REAL TemplateSqSum = SELECT_CHAN(templ_sqsum);

   // Compute the square sum of the area
   REAL v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);

   // Pre-computed cross correlation
   REAL ccorr = dest[gy * dst_step + gx];

   // Compute the normalized cross correlation
   dest[gy * dst_step + gx] = ccorr / sqrt(v * TemplateSqSum);
}

float2 complexMul(float2 a, float2 b)
{
   float2 c;
   c.x = a.x * b.x - a.y * b.y;
   c.y = a.x * b.y + a.y * b.x;
   return c;
}

kernel void copy_offset(global const TYPE * source, global float * dest, int src_step, int dst_step,
                        int offsetx, int offsety, int destw, int desth, int channel)
{
   src_step  /= sizeof(TYPE);
   dst_step  /= sizeof(float);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   int y = gy + offsety;
   if (y < 0)
      y += desth;

   int x = gx + offsetx;
   if (x < 0)
      x += destw;

   SCALAR v = GET_CHAN(source + gy * src_step + gx, channel);

   dest[y * dst_step + x] = convert_float(v);
}

kernel void copy_result(global const float * source, global TYPE * dest, int src_step, int dst_step,
                        int destw, int desth, int channel)
{
   src_step  /= sizeof(float);
   dst_step  /= sizeof(TYPE);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   if (gx > destw - 1 || gy > desth - 1)
      return;

   float v = source[gy * src_step + gx];

   SET_CHAN(dest + gy * dst_step + gx, channel, v);
}

kernel void mulAndScaleSpectrums(global const float2 * source, global const float2 * template,
                                 global float2 * dest, int src_step, int temp_step, int dst_step,
                                 int width, int height, float scale)
{
   src_step  /= sizeof(float2);
   temp_step /= sizeof(float2);
   dst_step  /= sizeof(float2);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
  
   float2 t = template[gy*temp_step + gx];

   t.y = -t.y;	// Conjugate

   float2 v = complexMul(source[gy*src_step + gx], t);

   v *= scale;

   dest[gy*dst_step + gx] = v;
}
