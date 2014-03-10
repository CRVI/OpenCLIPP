////////////////////////////////////////////////////////////////////////////////
//! @file	: Convert.cl
//! @date   : Jul 2013
//!
//! @brief  : Image depth conversion
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


#ifdef I

   // For signed integer images
   #define READ_IMAGE(img, pos) read_imagei(img, sampler, pos)
   #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, px)
   #define TYPE int4

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)
      #define TYPE uint4

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) read_imagef(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, px)
      #define TYPE float4

   #endif // UI

#endif // I


#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };


// Conversions
kernel void to_float(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imagef(dest, pos, convert_float4(src));
}

kernel void to_int(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imagei(dest, pos, convert_int4_sat(src));
}

kernel void to_uint(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imageui(dest, pos, convert_uint4_sat(src));
}

// Convert & scale
kernel void scale_to_float(read_only image2d_t source, write_only image2d_t dest, float offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos));

   // Write pixel
   write_imagef(dest, pos, src * ratio + offset);
}

kernel void scale_to_int(read_only image2d_t source, write_only image2d_t dest, int offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos)) * ratio;

   // Write pixel
   write_imagei(dest, pos, convert_int4_sat(src) + offset);
}

kernel void scale_to_uint(read_only image2d_t source, write_only image2d_t dest, int offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos)) * ratio;

   // Write pixel
   write_imageui(dest, pos, convert_uint4_sat(convert_int4_sat(src) + offset));
}

kernel void to_gray(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Average the first three channels into the first channel
   TYPE dst = (TYPE)((src.x + src.y + src.z) / 3, 0, 0, 0);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel1(read_only image2d_t source, write_only image2d_t dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the first channel
   TYPE dst = (TYPE)(src.x, src.x, src.x, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel2(read_only image2d_t source, write_only image2d_t dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the second channel
   TYPE dst = (TYPE)(src.y, src.y, src.y, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel3(read_only image2d_t source, write_only image2d_t dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the third channel
   TYPE dst = (TYPE)(src.z, src.z, src.z, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel4(read_only image2d_t source, write_only image2d_t dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the fourth channel
   TYPE dst = (TYPE)(src.w, src.w, src.w, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}
