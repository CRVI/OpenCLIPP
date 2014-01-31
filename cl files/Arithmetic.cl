////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.cl
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations on images
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

   #endif // UI

#endif // I


#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#define BINARY_OP(name, code) \
kernel void name(read_only image2d_t source1, read_only image2d_t source2, write_only image2d_t dest)\
{\
   BEGIN\
   float4 src1 = READ_IMAGE(source1, pos);\
   float4 src2 = READ_IMAGE(source2, pos);\
   WRITE_IMAGE(dest, pos, code);\
}

#define CONSTANT_OP(name, code) \
kernel void name(read_only image2d_t source, write_only image2d_t dest, float value)\
{\
   BEGIN\
   float4 src = READ_IMAGE(source, pos);\
   WRITE_IMAGE(dest, pos, code);\
}

#define UNARY_OP(name, code) \
kernel void name(read_only image2d_t source, write_only image2d_t dest)\
{\
   BEGIN\
   float4 src = READ_IMAGE(source, pos);\
   WRITE_IMAGE(dest, pos, code);\
}

// Mathematical - between two images
BINARY_OP(add_images, src1 + src2)
BINARY_OP(add_square_images, src1 + src2 * src2)
BINARY_OP(sub_images, src1 - src2)
BINARY_OP(abs_diff_images, fabs(src1 - src2))
BINARY_OP(mul_images, src1 * src2)
BINARY_OP(div_images, src1 / src2)
BINARY_OP(min_images, min(src1, src2))
BINARY_OP(max_images, max(src1, src2))
BINARY_OP(mean_images, (src1 + src2) * .5f)
BINARY_OP(combine_images, native_sqrt(src1 * src1 + src2 * src2))

// Mathematical - image and value
CONSTANT_OP(add_constant, src + value)
CONSTANT_OP(sub_constant, src - value)
CONSTANT_OP(abs_diff_constant, fabs(src - value))
CONSTANT_OP(mul_constant, src * value)
CONSTANT_OP(div_constant, src / value)
CONSTANT_OP(reversed_div, value / src)
CONSTANT_OP(min_constant, min(src, value))
CONSTANT_OP(max_constant, max(src, value))
CONSTANT_OP(mean_constant, (src + value) * .5f)

// Mathematical - calculation on one image
UNARY_OP(abs_image, fabs(src))
UNARY_OP(invert_image, 255.f - src)
UNARY_OP(sqr_image, src * src)

// float images recomended
UNARY_OP(exp_image, exp(src))
UNARY_OP(log_image, log(src))
UNARY_OP(sqrt_image, sqrt(src))
UNARY_OP(sin_image, sin(src))
UNARY_OP(cos_image, cos(src))
