////////////////////////////////////////////////////////////////////////////////
//! @file	: Lut.cl
//! @date   : Jul 2013
//!
//! @brief  : Lut transformation
//! 
//! Copyright (C) 2013 - CRVI
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

// Assumes vector size of VEC_WIDTH - must be called with img_type.VectorRange(VEC_WIDTH)
// Type must be specified when compiling this file, example : for unsigned 8 bit "-D U8"

#define VEC_WIDTH 4

#define BEGIN  \
   const int gx = get_global_id(0) * VEC_WIDTH;\
   const int gy = get_global_id(1);\
   src_step /= sizeof(SCALAR);\
   dst_step /= sizeof(SCALAR);


SCALAR do_lut(SCALAR input, constant const uint * levels, constant const uint * values, int nb)
{
   if (input < levels[0])
      return input;

   if (input >= levels[nb - 1])
      return input;

   int k = 0;
   while (k < nb - 1 && input >= levels[k + 1])
      k++;

   return values[k];
}

SCALAR do_lut_linear(SCALAR input, constant const float * levels, constant const float * values, int nb)
{
   if (input < levels[0])
      return input;

   if (input >= levels[nb - 1])
      return input;

   int k = 0;
   while (k < nb - 1 && input > levels[k])
      k++;

   if (k > 0)
      k--;

   float DiffL = levels[k + 1] - levels[k];
   float DiffV = values[k + 1] - values[k];
   float Diff2 = input - levels[k];
   float Diff = Diff2 / DiffL * DiffV;

   return values[k] + Diff;
}

kernel void lut(INPUT_SPACE const SCALAR * source, global SCALAR * dest, int src_step, int dst_step, int width,
                   constant const uint * levels, constant const uint * values, int nb)
{
   BEGIN

   if (VEC_WIDTH > 1 && gx + VEC_WIDTH > width)
   {
      for (int x = gx; x < width; x++)
         dest[gy * dst_step + x] = do_lut(source[gy * src_step + x], levels, values, nb);

      return;
   }

   for (int x = gx; x < gx + VEC_WIDTH; x++)
      dest[gy * dst_step + x] = do_lut(source[gy * src_step + x], levels, values, nb);
}

kernel void lut_linear(INPUT_SPACE const SCALAR * source, global SCALAR * dest, int src_step, int dst_step, int width,
                       constant const float * levels, constant const float * values, int nb)
{
   BEGIN

   if (VEC_WIDTH > 1 && gx + VEC_WIDTH > width)
   {
      for (int x = gx; x < width; x++)
         dest[gy * dst_step + x] = do_lut_linear(source[gy * src_step + x], levels, values, nb);

      return;
   }

   for (int x = gx; x < gx + VEC_WIDTH; x++)
      dest[gy * dst_step + x] = do_lut_linear(source[gy * src_step + x], levels, values, nb);
}

#ifndef FLOAT
// Optimized LUT : All values in source image must be between 0 and 255, there must be excatly 256 levels
kernel void lut_256(INPUT_SPACE const SCALAR * source, global SCALAR * dest, int src_step, int dst_step, int width,
                   constant const uchar * levels)
{
   BEGIN

   if (VEC_WIDTH > 1 && gx + VEC_WIDTH > width)
   {
      for (int x = gx; x < width; x++)
         dest[gy * dst_step + x] = levels[source[gy * src_step + x]];

      return;
   }

   for (int x = gx; x < gx + VEC_WIDTH; x++)
      dest[gy * dst_step + x] = levels[source[gy * src_step + x]];
}

__attribute__((reqd_work_group_size(16, 16, 1)))
kernel void lut_256_cached(INPUT_SPACE const SCALAR * source, global SCALAR * dest, int src_step, int dst_step,
                   constant const uchar * levels)
{
   BEGIN
      
   // Do a local cache of the levels to improve speed
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);
   local uchar levels_local[256];
   levels_local[lid] = levels[lid];
   barrier(CLK_LOCAL_MEM_FENCE);

   for (int x = gx; x < gx + VEC_WIDTH; x++)
      dest[gy * dst_step + x] = levels_local[source[gy * src_step + x]];
}
#endif // FLOAT
