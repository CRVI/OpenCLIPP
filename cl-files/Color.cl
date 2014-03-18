////////////////////////////////////////////////////////////////////////////////
//! @file	: Color.cl
//! @date   : Jul 2013
//!
//! @brief  : 3 Channel & 4 Channel image conversion
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


constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

// Type must be specified when compiling this file, example : for unsigned 8 bit "-D U8"

#ifdef S8
#define TYPE char
#endif

#ifdef U8
#define TYPE uchar
#define UNSIGNED
#endif

#ifdef S16
#define TYPE short
#endif

#ifdef U16
#define TYPE ushort
#define UNSIGNED
#endif

#ifdef S32
#define TYPE int
#endif

#ifdef U32
#define TYPE uint
#define UNSIGNED
#endif

#ifdef F32
#define TYPE float
#define FLOAT
#endif

#ifndef TYPE
#define TYPE uchar
#define UNSIGNED
#endif


#ifdef FLOAT
   // For float
   #define PIXEL float4
   #define READ_IMAGE(img, pos) read_imagef(img, sampler, pos)
   #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, px)
#else // FLOAT
   #ifdef UNSIGNED
      // For unsigned integer images
      #define PIXEL uint4
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)
   #else // UNSIGNED
      // For signed integer images
      #define PIXEL int4
      #define READ_IMAGE(img, pos) read_imagei(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, px)
   #endif // UNSIGNED
#endif // FLOAT

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

kernel void Convert3CTo4C(global const TYPE * source, write_only image2d_t dest, uint source_step)
{
   BEGIN

   source_step /= sizeof(TYPE);

   int source_index = gx * 3 + gy * source_step;

   PIXEL color;
   color.x = source[source_index + 0];
   color.y = source[source_index + 1];
   color.z = source[source_index + 2];
   color.w = 255;

   // Write pixel
   WRITE_IMAGE(dest, pos, color);
}

kernel void Convert4CTo3C(read_only image2d_t source, global TYPE * dest, uint dest_step)
{
   BEGIN

   dest_step /= sizeof(TYPE);

   // Read pixel
   PIXEL color = READ_IMAGE(source, pos);

   // Write pixel
   int dest_index = gx * 3 + gy * dest_step;
   dest[dest_index + 0] = color.x;
   dest[dest_index + 1] = color.y;
   dest[dest_index + 2] = color.z;
}
