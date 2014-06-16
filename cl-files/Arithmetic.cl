////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.cl
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations on image buffers
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

#define INTERNAL_SCALAR float // Use float for intermediary values

#include "Vector.h"

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
BINARY_OP(combine, native_sqrt(src1 * src1 + src2 * src2))


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
UNARY_OP(exp_image, exp(src))
UNARY_OP(log_image, log(src))
UNARY_OP(sqrt_image, sqrt(src))
UNARY_OP(sin_image, sin(src))
UNARY_OP(cos_image, cos(src))


// Un-macroed version - For debugging purposes
/*kernel void add_constant(global const ushort * source, global ushort * dest, int src_step, int dst_step, int width, float value)
{
   const int gx = get_global_id(0);	// x divided by VEC_WIDTH
   const int gy = get_global_id(1);
   src_step /= sizeof(ushort);
   dst_step /= sizeof(ushort);

   float value = value_arg;

   //if (gx != 0 || gy != 0)
   //   return;

   if ((gx + 1) * 4 > width)
   {
      // Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH
      for (int i = gx * 4; i < width; i++)
      {
         const float src = convert_float(source[(gy * src_step) + i]);
         dest[(gy * dst_step) + i] = convert_ushort_sat(src + value);
      }
      return;
   }

   if (Unaligned)
   {
      for (int i = gx * VEC_WIDTH; i < (gx + 1) * VEC_WIDTH; i++)
      {
         const float src = convert_float(source[(gy * src_step) + i]);
         dest[(gy * dst_step) + i] = convert_ushort_sat(src + value);
      }
      return;
   }

   const float4 src = convert_float4(*(const global ushort4 *)(source + gy * src_step + gx * VEC_WIDTH));
   *(global ushort4 *)(img + gy * dst_step + gx * VEC_WIDTH) = convert_ushort4_sat(src + value);
}

kernel void add_constant_flush(global const ushort * source, global ushort * dest, int src_step, int dst_step, int width, float value)
{
   const int gx = get_global_id(0);	// x divided by VEC_WIDTH
   const int gy = get_global_id(1);
   src_step /= sizeof(ushort);
   dst_step /= sizeof(ushort);

   float value = value_arg;

   //if (gx != 0 || gy != 0)
   //   return;

   const float4 src = convert_float4(*(const global ushort4 *)(source + gx * VEC_WIDTH));
   *(global ushort4 *)(img + gx * VEC_WIDTH) = convert_ushort4_sat(src + value);
}*/
