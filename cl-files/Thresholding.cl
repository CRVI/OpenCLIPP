////////////////////////////////////////////////////////////////////////////////
//! @file	: Threshold.cl
//! @date   : Jul 2013
//!
//! @brief  : Image thresholding
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
      #define FLOAT

   #endif // UI

#endif // I

#define DST_TYPE TYPE

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };


kernel void thresholdLT(read_only image2d_t source, write_only image2d_t dest, float thresh, float valueLower)
{
   BEGIN

   // Read pixel
   TYPE color = READ_IMAGE(source, pos);

   // Modify color
   TYPE dst_color = (color < thresh ? (TYPE)(valueLower, valueLower, valueLower, valueLower) : color);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst_color);
}

kernel void thresholdGT(read_only image2d_t source, write_only image2d_t dest, float thresh, float valueHigher)
{
   BEGIN

   // Read pixel
   TYPE color = READ_IMAGE(source, pos);

   // Modify color
   TYPE dst_color = (color > thresh ? (TYPE)(valueHigher, valueHigher, valueHigher, valueHigher) : color);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst_color);
}

kernel void thresholdGTLT(read_only image2d_t source, write_only image2d_t dest, float threshLT,
                         float valueLower, float threshGT, float valueHigher)
{
   BEGIN

   // Read pixel
   TYPE src_color = READ_IMAGE(source, pos);

   // Modify color
   TYPE dst_color = (src_color < threshLT ? (TYPE)(valueLower, valueLower, valueLower, valueLower) : src_color);
   dst_color = (src_color > threshGT ? (TYPE)(valueHigher, valueHigher, valueHigher, valueHigher) : dst_color);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst_color);
}

#define BINARY_OP(name, code)\
kernel void name(read_only image2d_t source1, read_only image2d_t source2, write_only image2d_t dest)\
{\
   BEGIN\
   TYPE src1 = READ_IMAGE(source1, pos);\
   TYPE src2 = READ_IMAGE(source2, pos);\
   DST_TYPE dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}

#define CONSTANT_OP(name, code)\
kernel void name(read_only image2d_t source, write_only image2d_t dest, float value)\
{\
   BEGIN\
   TYPE src = READ_IMAGE(source, pos);\
   DST_TYPE dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}


BINARY_OP(img_thresh_LT, (src1 < src2 ? src1 : src2))
BINARY_OP(img_thresh_LQ, (src1 <= src2 ? src1 : src2))
BINARY_OP(img_thresh_EQ, (src1 == src2 ? src1 : src2))
BINARY_OP(img_thresh_GQ, (src1 >= src2 ? src1 : src2))
BINARY_OP(img_thresh_GT, (src1 > src2 ? src1 : src2))


#define WHITE ((uint4)(255, 255, 255, 0))
#define BLACK ((uint4)(0, 0, 0, 0))

#undef WRITE_IMAGE
#define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)

#define DST_TYPE uint4

BINARY_OP(img_compare_LT, (src1 < src2 ? WHITE : BLACK))
BINARY_OP(img_compare_LQ, (src1 <= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_EQ, (src1 == src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GQ, (src1 >= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GT, (src1 > src2 ? WHITE : BLACK))

CONSTANT_OP(compare_LT, (src < value ? WHITE : BLACK))
CONSTANT_OP(compare_LQ, (src <= value ? WHITE : BLACK))
CONSTANT_OP(compare_EQ, (src == value ? WHITE : BLACK))
CONSTANT_OP(compare_GQ, (src >= value ? WHITE : BLACK))
CONSTANT_OP(compare_GT, (src > value ? WHITE : BLACK))
