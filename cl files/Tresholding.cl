////////////////////////////////////////////////////////////////////////////////
//! @file	: Tresholding.cl
//! @date   : Jul 2013
//!
//! @brief  : Image tresholding
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
   #define READ_IMAGE(img, pos) convert_float4(read_imagei(img, sampler, pos))
   #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, convert_int4_sat(px))

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) convert_float4(read_imageui(img, sampler, pos))
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, convert_uint4_sat(px))

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) read_imagef(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, px)
      #define TYPE float4
      #define SCALAR float
      #define FLOAT

   #endif // UI

#endif // I


#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };


kernel void tresholdLT(read_only image2d_t source, write_only image2d_t dest, float thresh, float valueLower)
{
   BEGIN

   // Read pixel
   float4 color = READ_IMAGE(source, pos);

   // Modify color
   if (color.x < thresh)
      color = (float4)(valueLower, valueLower, valueLower, valueLower);

   // Write pixel
   WRITE_IMAGE(dest, pos, color);
}

kernel void tresholdGT(read_only image2d_t source, write_only image2d_t dest, float thresh, float valueHigher)
{
   BEGIN

   // Read pixel
   float4 color = READ_IMAGE(source, pos);

   // Modify color
   if (color.x > thresh)
      color = (float4)(valueHigher, valueHigher, valueHigher, valueHigher);

   // Write pixel
   WRITE_IMAGE(dest, pos, color);
}

kernel void tresholdGTLT(read_only image2d_t source, write_only image2d_t dest, float threshLT,
                         float valueLower, float treshGT, float valueHigher)
{
   BEGIN

   // Read pixel
   float4 src_color = READ_IMAGE(source, pos);

   float4 dst_color = src_color;

   // Modify color
   if (src_color.x < threshLT)
      dst_color = (float4)(valueLower, valueLower, valueLower, valueLower);

   if (src_color.x > treshGT)
      dst_color = (float4)(valueHigher, valueHigher, valueHigher, valueHigher);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst_color);
}

#define BINARY_OP(name, code)\
kernel void name(read_only image2d_t source1, read_only image2d_t source2, write_only image2d_t dest)\
{\
   BEGIN\
   float4 src1 = READ_IMAGE(source1, pos);\
   float4 src2 = READ_IMAGE(source1, pos);\
   float4 dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}

#define CONSTANT_OP(name, code)\
kernel void name(read_only image2d_t source, write_only image2d_t dest, float value)\
{\
   BEGIN\
   float4 src = READ_IMAGE(source, pos);\
   float4 dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}

BINARY_OP(img_tresh_LT, (src1 < src2 ? src1 : src2))
BINARY_OP(img_tresh_LQ, (src1 <= src2 ? src1 : src2))
BINARY_OP(img_tresh_EQ, (src1 == src2 ? src1 : src2))
BINARY_OP(img_tresh_GQ, (src1 >= src2 ? src1 : src2))
BINARY_OP(img_tresh_GT, (src1 > src2 ? src1 : src2))

BINARY_OP(img_compare_LT, (src1.x < src2.x))
BINARY_OP(img_compare_LQ, (src1.x <= src2.x))
BINARY_OP(img_compare_EQ, (src1.x == src2.x))
BINARY_OP(img_compare_GQ, (src1.x >= src2.x))
BINARY_OP(img_compare_GT, (src1.x > src2.x))

CONSTANT_OP(compare_LT, (src.x < value))
CONSTANT_OP(compare_LQ, (src.x <= value))
CONSTANT_OP(compare_EQ, (src.x == value))
CONSTANT_OP(compare_GQ, (src.x >= value))
CONSTANT_OP(compare_GT, (src.x > value))
