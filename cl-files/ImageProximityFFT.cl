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

#define WRITE(img, val) img[gy * dst_step + gx] = val

float AreaSqSum(INPUT_SPACE const SCALAR * SqSumImage, int src_step, int w, int h, int dest_width, int dest_height)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);	

   int Left   = gx     - w / 2 - 1;
   int Right  = gx + w - w / 2 - 1;
   int Top    = gy     - h / 2 - 1;
   int Bottom = gy + h - h / 2 - 1;
   
   Right  = clamp(Right,  0, dest_width  - 1);
   Bottom = clamp(Bottom, 0, dest_height - 1);

   float TopLeft = 0;
   float TopRight = 0;
   float BottomLeft = 0;

   if (Top >= 0 && Left >= 0)
      TopLeft = SqSumImage[Top * src_step + Left];

   if (Top >= 0)
      TopRight = SqSumImage[Top * src_step + Right];
   
   if (Left >= 0)
      BottomLeft = SqSumImage[Bottom * src_step + Left];

   float BottomRight = SqSumImage[Bottom * src_step + Right];

   return (BottomRight - TopRight) - (BottomLeft - TopLeft);
}

kernel void matchTemplatePreparedSQDIFF(INPUT_SPACE const float * source, global float * dest, int w, int h, float templ_sqsum, int src_step, int dst_step, int dest_width, int dest_height)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step  /= sizeof(SCALAR);
   dst_step  /= sizeof(float);

   float v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);
   float ccorr = dest[gy * dst_step + gx];
   dest[gy * dst_step + gx] = v - 2.f * ccorr + templ_sqsum;
}

kernel void matchTemplatePreparedSQDIFF_NORM(INPUT_SPACE const float * source, global float * dest, int w, int h, float templ_sqsum, int src_step, int dst_step, int dest_width, int dest_height)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step  /= sizeof(SCALAR);
   dst_step  /= sizeof(float);

   float v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);
   float ccorr = dest[gy * dst_step + gx];
   dest[gy * dst_step + gx] = (v - 2.f * ccorr + templ_sqsum)/sqrt(v * templ_sqsum);
}

kernel void matchTemplatePreparedCCORR_NORM(INPUT_SPACE const float * source, global float * dest, int w, int h, float templ_sqsum, int src_step, int dst_step, int dest_width, int dest_height)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   src_step  /= sizeof(SCALAR);
   dst_step  /= sizeof(float);

   float v = AreaSqSum(source, src_step, w, h, dest_width, dest_height);
   float ccorr = dest[gy * dst_step + gx];
   dest[gy * dst_step + gx] = ccorr/sqrt(v * templ_sqsum);
}

float2 complexMul(float2 a, float2 b)
{
   float2 c;
   c.x = a.x * b.x - a.y * b.y;
   c.y = a.x * b.y + a.y * b.x;
   return c;
}

kernel void copy_offset(global const SCALAR * source, global float * dest, int src_step, int dst_step, int offsetx, int offsety, int destw, int desth)
{
   src_step  /= sizeof(SCALAR);
   dst_step  /= sizeof(float);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   int y = gy + offsety;
   if (y < 0)
      y += desth;

   int x = gx + offsetx;
   if (x < 0)
      x += destw;

   dest[y * dst_step + x] = (float)source[gy * src_step + gx];
}

kernel void copy_roi(global const TYPE * source, global TYPE * dest, int src_step, int dst_step, int destw, int desth)
{
   src_step  /= sizeof(TYPE);
   dst_step  /= sizeof(TYPE);
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   if (gx > destw - 1 || gy > desth - 1)
      return;

   dest[gy * dst_step + gx] = source[gy * src_step + gx];
}

kernel void mulAndScaleSpectrums(global const float2 * source, global const float2 * template, global float2 * dest, int src_step, int temp_step, int dst_step, int width, int height, float scale)
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

