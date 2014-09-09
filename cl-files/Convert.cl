////////////////////////////////////////////////////////////////////////////////
//! @file	: Convert.cl
//! @date   : Jun 2014
//!
//! @brief  : Image depth conversion
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

#define INPUT INPUT_SPACE const TYPE *

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#ifdef NBCHAN
#if NBCHAN >= 2
kernel void select_channel1(INPUT source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE src = source[gy * src_step + gx];

   // Select the first channel
   dest[gy * dst_step + gx] = src.x;
}

kernel void select_channel2(INPUT source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE src = source[gy * src_step + gx];

   // Select the second channel
   dest[gy * dst_step + gx] = src.y;
}

#endif   // NBCHAN >= 2


#if NBCHAN == 4

kernel void to_gray(INPUT source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE src = source[gy * src_step + gx];

   // Select the first channel
   dest[gy * dst_step + gx] = (src.x + src.y + src.z) / 3;
}

kernel void select_channel3(INPUT source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE src = source[gy * src_step + gx];

   // Select the third channel
   dest[gy * dst_step + gx] = src.z;
}

kernel void select_channel4(INPUT source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE src = source[gy * src_step + gx];

   // Select the fourth channel
   dest[gy * dst_step + gx] = src.w;
}

#endif   // NBCHAN == 4
#endif   // NBCHAN

#ifndef NBCHAN
#define NBCHAN
#endif

#define CONVERT_KERNEL(name, dest_type) \
   kernel void name(INPUT source, global dest_type * dest, int src_step, int dst_step)\
   {\
      BEGIN\
      src_step /= sizeof(TYPE);\
      dst_step /= sizeof(dest_type);\
      TYPE src = source[gy * src_step + gx];\
      dest[gy * dst_step + gx] = CONCATENATE(convert_, CONCATENATE(dest_type, _sat)) (src);\
   }

#define CONVERT_FLOAT_KERNEL(name, dest_type) \
   kernel void name(INPUT source, global dest_type * dest, int src_step, int dst_step)\
   {\
      BEGIN\
      src_step /= sizeof(TYPE);\
      dst_step /= sizeof(dest_type);\
      TYPE src = source[gy * src_step + gx];\
      dest[gy * dst_step + gx] = CONCATENATE(convert_, dest_type) (src);\
   }

#define SCALE_KERNEL(name, dest_type) \
   kernel void name(INPUT source, global dest_type * dest, int src_step, int dst_step, float offset, float ratio)\
   {\
      BEGIN\
      src_step /= sizeof(TYPE);\
      dst_step /= sizeof(dest_type);\
      TYPE src = source[gy * src_step + gx];\
      dest[gy * dst_step + gx] = \
         CONCATENATE(convert_, CONCATENATE(dest_type, _sat)) (CONVERT_REAL(src) * ratio + offset );\
   }

#define SCALE_FLOAT_KERNEL(name, dest_type) \
   kernel void name(INPUT source, global dest_type * dest, int src_step, int dst_step, float offset, float ratio)\
   {\
      BEGIN\
      src_step /= sizeof(TYPE);\
      dst_step /= sizeof(dest_type);\
      TYPE src = source[gy * src_step + gx];\
      dest[gy * dst_step + gx] = CONCATENATE(convert_, dest_type) (CONVERT_REAL(src) * ratio + offset);\
   }

CONVERT_KERNEL(to_uchar,   CONCATENATE(uchar,  NBCHAN))
CONVERT_KERNEL(to_char,    CONCATENATE(char,   NBCHAN))
CONVERT_KERNEL(to_ushort,  CONCATENATE(ushort, NBCHAN))
CONVERT_KERNEL(to_short,   CONCATENATE(short,  NBCHAN))
CONVERT_KERNEL(to_uint,    CONCATENATE(uint,   NBCHAN))
CONVERT_KERNEL(to_int,     CONCATENATE(int,    NBCHAN))
CONVERT_FLOAT_KERNEL(to_float,  CONCATENATE(float,  NBCHAN))
CONVERT_FLOAT_KERNEL(to_double, CONCATENATE(double, NBCHAN))

SCALE_KERNEL(scale_to_uchar,  CONCATENATE(uchar,  NBCHAN))
SCALE_KERNEL(scale_to_char,   CONCATENATE(char,   NBCHAN))
SCALE_KERNEL(scale_to_ushort, CONCATENATE(ushort, NBCHAN))
SCALE_KERNEL(scale_to_short,  CONCATENATE(short,  NBCHAN))
SCALE_KERNEL(scale_to_uint,   CONCATENATE(uint,   NBCHAN))
SCALE_KERNEL(scale_to_int,    CONCATENATE(int,    NBCHAN))
SCALE_FLOAT_KERNEL(scale_to_float,  CONCATENATE(float,  NBCHAN))
SCALE_FLOAT_KERNEL(scale_to_double, CONCATENATE(double, NBCHAN))

kernel void to_2channels(INPUT_SPACE SCALAR * source, global TYPE2 * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(TYPE2);
   SCALAR src = source[gy * src_step + gx];
   dest[gy * dst_step + gx] = (TYPE2)(src, src);
}

kernel void to_3channels(INPUT_SPACE SCALAR * source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(SCALAR);
   dst_step /= (sizeof(SCALAR) * 3);
   SCALAR src = source[gy * src_step + gx];

   int dest_index = gx * 3 + gy * dst_step;
   dest[dest_index + 0] = src;
   dest[dest_index + 1] = src;
   dest[dest_index + 2] = src;
}

kernel void to_4channels(INPUT_SPACE SCALAR * source, global TYPE4 * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(TYPE4);
   SCALAR src = source[gy * src_step + gx];
   dest[gy * dst_step + gx] = (TYPE4)(src, src, src, 255);
}

kernel void copy(INPUT source, global TYPE * dest, int src_step, int dst_step)
{
   BEGIN
   src_step /= sizeof(TYPE);
   dst_step /= sizeof(TYPE);
   dest[gy * dst_step + gx] = source[gy * src_step + gx];
}

kernel void copy3Cto4C(INPUT_SPACE SCALAR * source, global TYPE4 * dest, int src_step, int dst_step)
{
   BEGIN

   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(TYPE4);

   int source_index = gx * 3 + gy * src_step;

   TYPE4 color;
   color.x = source[source_index + 0];
   color.y = source[source_index + 1];
   color.z = source[source_index + 2];
   color.w = 255;

   // Write pixel
   dest[gy * dst_step + gx] = color;
}

kernel void copy4Cto3C(INPUT_SPACE TYPE4 * source, global SCALAR * dest, int src_step, int dst_step)
{
   BEGIN

   src_step /= sizeof(TYPE4);
   dst_step /= sizeof(SCALAR);

   // Read pixel
   TYPE4 color = source[gy * src_step + gx];

   // Write pixel
   int dest_index = gx * 3 + gy * dst_step;
   dest[dest_index + 0] = color.x;
   dest[dest_index + 1] = color.y;
   dest[dest_index + 2] = color.z;
}
