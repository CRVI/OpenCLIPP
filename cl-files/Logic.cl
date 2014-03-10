////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.cl
//! @date   : Jul 2013
//!
//! @brief  : Logic (bitwise) operations on images
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
   #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, mask_result(px, img))
   #define TYPE int4

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, mask_result(px, img))
      #define TYPE uint4

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) convert_int4_sat(read_imagef(img, sampler, pos))
      #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, convert_float4(px))
      #define TYPE int4

   #endif // UI

#endif // I

int get_mask(write_only image2d_t img)
{
   int type = get_image_channel_data_type(img);

   switch (type)
   {
   case CLK_SIGNED_INT8:
   case CLK_UNSIGNED_INT8:
      return 0xFF;
   case CLK_SIGNED_INT16:
   case CLK_UNSIGNED_INT16:
      return 0xFFFF;
   }

   return 0xFFFFFFFF;
}

TYPE mask_result(TYPE val, write_only image2d_t img)
{
   int mask = get_mask(img);
   return val & mask;
}


#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#define BINARY_OP(name, code) \
kernel void name(read_only image2d_t source1, read_only image2d_t source2, write_only image2d_t dest)\
{\
   BEGIN\
   TYPE src1 = READ_IMAGE(source1, pos);\
   TYPE src2 = READ_IMAGE(source2, pos);\
   WRITE_IMAGE(dest, pos, code);\
}

#define CONSTANT_OP(name, type, code) \
kernel void name(read_only image2d_t source, write_only image2d_t dest, type value_in)\
{\
   BEGIN\
   TYPE value = {value_in, value_in, value_in, value_in};\
   TYPE src = READ_IMAGE(source, pos);\
   WRITE_IMAGE(dest, pos, code);\
}

#define UNARY_OP(name, code) \
kernel void name(read_only image2d_t source, write_only image2d_t dest)\
{\
   BEGIN\
   TYPE src = READ_IMAGE(source, pos);\
   WRITE_IMAGE(dest, pos, code);\
}


// Bitwise operations

// between two images
BINARY_OP(and_images, src1 & src2)
BINARY_OP(or_images, src1 | src2)
BINARY_OP(xor_images, src1 ^ src2)

// image and value
CONSTANT_OP(and_constant, uint, src & value)
CONSTANT_OP(or_constant, uint, src | value)
CONSTANT_OP(xor_constant, uint, src ^ value)

// Unary
UNARY_OP(not_image, ~src)
