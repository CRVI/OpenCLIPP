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

#include "Images.h"


// Conversions
kernel void to_float(INPUT source, OUTPUT dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imagef(dest, pos, convert_float4(src));
}

kernel void to_int(INPUT source, OUTPUT dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imagei(dest, pos, convert_int4_sat(src));
}

kernel void to_uint(INPUT source, OUTPUT dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Write pixel
   write_imageui(dest, pos, convert_uint4_sat(src));
}

// Convert & scale
kernel void scale_to_float(INPUT source, OUTPUT dest, float offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos));

   // Write pixel
   write_imagef(dest, pos, src * ratio + offset);
}

kernel void scale_to_int(INPUT source, OUTPUT dest, int offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos)) * ratio;

   // Write pixel
   write_imagei(dest, pos, convert_int4_sat(src) + offset);
}

kernel void scale_to_uint(INPUT source, OUTPUT dest, int offset, float ratio)
{
   BEGIN

   // Read pixel
   float4 src = convert_float4(READ_IMAGE(source, pos)) * ratio;

   // Write pixel
   write_imageui(dest, pos, convert_uint4_sat(convert_int4_sat(src) + offset));
}

kernel void to_gray(INPUT source, OUTPUT dest)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Average the first three channels into the first channel
   TYPE dst = (TYPE)((src.x + src.y + src.z) / 3, 0, 0, 0);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel1(INPUT source, OUTPUT dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the first channel
   TYPE dst = (TYPE)(src.x, src.x, src.x, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel2(INPUT source, OUTPUT dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the second channel
   TYPE dst = (TYPE)(src.y, src.y, src.y, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel3(INPUT source, OUTPUT dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the third channel
   TYPE dst = (TYPE)(src.z, src.z, src.z, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}

kernel void select_channel4(INPUT source, OUTPUT dest, int channel_no)
{
   BEGIN

   // Read pixel
   TYPE src = READ_IMAGE(source, pos);

   // Select the fourth channel
   TYPE dst = (TYPE)(src.w, src.w, src.w, 255);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst);
}
