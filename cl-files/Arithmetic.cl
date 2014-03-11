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

#include "Images.h"


#define BINARY_OP(name, code) \
kernel void name(INPUT source1, INPUT source2, OUTPUT dest)\
{\
   BEGIN\
   float4 src1 = CONVERT_FLOAT(READ_IMAGE(source1, pos));\
   float4 src2 = CONVERT_FLOAT(READ_IMAGE(source2, pos));\
   WRITE_IMAGE(dest, pos, CONVERT(code));\
}

#define CONSTANT_OP(name, code) \
kernel void name(INPUT source, OUTPUT dest, float value)\
{\
   BEGIN\
   float4 src = CONVERT_FLOAT(READ_IMAGE(source, pos));\
   WRITE_IMAGE(dest, pos, CONVERT(code));\
}

#define UNARY_OP(name, code) \
kernel void name(INPUT source, OUTPUT dest)\
{\
   BEGIN\
   float4 src = CONVERT_FLOAT(READ_IMAGE(source, pos));\
   WRITE_IMAGE(dest, pos, CONVERT(code));\
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
