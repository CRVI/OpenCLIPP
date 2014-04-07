////////////////////////////////////////////////////////////////////////////////
//! @file	: Vectors.h
//! @date   : Mar 2014
//!
//! @brief  : Macros for vectorized operations
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

// Type must be specified when compiling this file, example : for unsigned 8 bit "-D U8 -D VEC_WIDTH=8"

// Optimization note : On my GTX 680 - fastest version for 8U is with no WITH_PADDING and VEC_WIDTH==8

// VEC_WIDTH is the width of vector operations
// It must be specified when compiling
#ifndef VEC_WIDTH
#define VEC_WIDTH 8    // Number of items done in parralel per worker - Can be 2, 4, 8 or 16
#endif


// ARG_TYPE is the type of constant arguments - defaults to float
#ifndef ARG_TYPE
#define ARG_TYPE float
#endif


#define TYPE CONCATENATE(SCALAR, VEC_WIDTH)  // Example : uchar8


// INTERNAL & INTERNAL_SCALAR are the types used for the calculations
// if INTERNAL_SCALAR is not defined when this file is included,
// INTERNAL_SCALAR will be the same type as the input image type
#ifndef INTERNAL_SCALAR
#define INTERNAL_SCALAR SCALAR
#define INTERNAL_SAME_TYPE
#endif

#define INTERNAL CONCATENATE(INTERNAL_SCALAR, VEC_WIDTH)

#ifdef INTERNAL_SAME_TYPE
#define CONVERT_INTERNAL(val) val
#define CONVERT_INTERNAL_SCALAR(val) val
#else
#define CONVERT_INTERNAL(val) CONCATENATE(convert_, INTERNAL) (val)
#define CONVERT_INTERNAL_SCALAR(val) CONCATENATE(convert_, INTERNAL_SCALAR) (val)
#endif


// DST & DST_SCALAR are the type of the destination image
// if DST_SCALAR is not defined when this file is included,
// DST_SCALAR will be the same type as the input image type
// DST_SCALAR is usually the same type as the input image
#ifndef DST_SCALAR
#define DST_SCALAR SCALAR
#ifdef FLOAT
#define DST_FLOAT
#endif
#endif

#define DST CONCATENATE(DST_SCALAR, VEC_WIDTH)

#ifdef DST_FLOAT
#define CONVERT_DST(val) CONCATENATE(convert_, DST) (val)
#define CONVERT_DST_SCALAR(val) CONCATENATE(convert_, DST_SCALAR) (val)
#else
#define CONVERT_DST(val) CONCATENATE(CONCATENATE(convert_, DST), _sat) (val)
#define CONVERT_DST_SCALAR(val) CONCATENATE(CONCATENATE(convert_, DST_SCALAR), _sat) (val)
#endif



#define BEGIN  \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src_step /= sizeof(SCALAR);\
   dst_step /= sizeof(DST_SCALAR);

#define BEGIN2 \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src1_step /= sizeof(SCALAR);\
   src2_step /= sizeof(SCALAR);\
   dst_step /= sizeof(DST_SCALAR);


//#define WITH_PADDING  // Is defined by runtime when needed

#ifdef WITH_PADDING
// This versions is capable of handling all types of image, including :
//    Width that is not a multiple of VEC_WIDTH
//    Images with step larger than Width * channels (ROI or Images with padding)
// If using a small ROI on a big image, this version will be faster
// must be called with img_type.VectorRange(VEC_WIDTH)

#define READ_IMAGE(img, step, x, y) CONVERT_INTERNAL(*(const INPUT_SPACE TYPE *)(img + y * step + x * VEC_WIDTH))
#define WRITE_IMAGE(img, step, x, y, val) *(global DST *)(img + y * step + x * VEC_WIDTH) = CONVERT_DST(val)

#define READ_SCALAR(img, step, x, y) CONVERT_INTERNAL_SCALAR(img[(y * step) + (x)])
#define WRITE_SCALAR(img, step, x, y, val) img[(y * step) + (x)] = CONVERT_DST_SCALAR(val)


#define PREPARE_SCALAR(x) \
   const INTERNAL_SCALAR src = READ_SCALAR(source, src_step, x, gy);

#define PREPARE_SCALAR2(x) \
   const INTERNAL_SCALAR src1 = READ_SCALAR(source1, src1_step, x, gy);\
   const INTERNAL_SCALAR src2 = READ_SCALAR(source2, src2_step, x, gy);

#define SCALAR_OP(code) WRITE_SCALAR(dest, dst_step, i, gy, (code))

//#define UNALIGNED    // Is defined by runtime when needed

#ifdef UNALIGNED

// If the step of the image is not a multiple of VEC_WIDTH * sizeof(SCALAR),
// accessing the pixels using a pointer to a vector type will not work (will either read the wrong pixels or will have an undefied behaviour).
// See https://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/dataTypes.html
// This code operates on the image using scalar operations and is significantly slower

#define UNALIGNED_CODE(code) \
   for (int i = gx * VEC_WIDTH; i < (gx + 1) * VEC_WIDTH; i++)\
   {\
      PREPARE_SCALAR(i)\
      SCALAR_OP(code);\
   }\
   return;

#define UNALIGNED_CODE2(code) \
   for (int i = gx * VEC_WIDTH; i < (gx + 1) * VEC_WIDTH; i++)\
   {\
      PREPARE_SCALAR2(i)\
      SCALAR_OP(code);\
   }\
   return;

#else

// For normal images, the step is a multiple of VEC_WIDTH * sizeof(SCALAR) and we can use the vector operations

#define UNALIGNED_CODE(code)
#define UNALIGNED_CODE2(code)

#endif // UNALIGNED

#define SCALAR_CODE(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR(i)\
         SCALAR_OP(code);\
      }\
      return;\
   }\
   UNALIGNED_CODE(code)

#define SCALAR_CODE2(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR2(i)\
         SCALAR_OP(code);\
      }\
      return;\
   }\
   UNALIGNED_CODE2(code)

#else // WITH_PADDING

// This version handles "flush" images, it is simpler and a bit faster.
// It only handles images that have :
//    Width that is a multiple of VEC_WIDTH
//       Improper width will cause the last few pixels not to be processed
//    Step == Width * Channels * Depth / 8 (no ROI and no padding at the end of the line)
//       All data between the image lines will be processed
// This version needs a 1D Range : cl::NDRange(Width * Height * Channels / VEC_WIDTH, 1, 1)

#define READ_IMAGE(img, step, x, y) CONVERT_INTERNAL(*(const INPUT_SPACE TYPE *)(img + x * VEC_WIDTH))
#define WRITE_IMAGE(img, step, x, y, val) *(global DST *)(img + x * VEC_WIDTH) = CONVERT_DST((val))

#define SCALAR_CODE(code)
#define SCALAR_CODE2(code)

#endif   // WITH_PADDING


#define PREPARE_VECTOR \
   const INTERNAL src = READ_IMAGE(source, src_step, gx, gy);

#define PREPARE_VECTOR2 \
   const INTERNAL src1 = READ_IMAGE(source1, src1_step, gx, gy);\
   const INTERNAL src2 = READ_IMAGE(source2, src2_step, gx, gy);

#define VECTOR_OP(code) WRITE_IMAGE(dest, dst_step, gx, gy, (code))


#define BINARY_OP(name, code) \
kernel void name(INPUT_SPACE const SCALAR * source1, INPUT_SPACE const SCALAR * source2,\
                global DST_SCALAR * dest, int src1_step, int src2_step, int dst_step, int width)\
{\
   BEGIN2\
   SCALAR_CODE2(code)\
   PREPARE_VECTOR2\
   VECTOR_OP(code);\
}

#define CONSTANT_OP(name, code) \
kernel void name(INPUT_SPACE const SCALAR * source, global DST_SCALAR * dest, int src_step, int dst_step, int width, ARG_TYPE value_arg)\
{\
   BEGIN\
   INTERNAL_SCALAR value = value_arg;\
   SCALAR_CODE(code)\
   PREPARE_VECTOR\
   VECTOR_OP(code);\
}

#define UNARY_OP(name, code) \
kernel void name(INPUT_SPACE const SCALAR * source, global DST_SCALAR * dest, int src_step, int dst_step, int width)\
{\
   BEGIN\
   SCALAR_CODE(code)\
   PREPARE_VECTOR\
   VECTOR_OP(code);\
}

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
