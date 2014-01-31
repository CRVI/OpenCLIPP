////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_Logic.cl
//! @date   : Jul 2013
//!
//! @brief  : Logic (bitwise) operations on image buffers
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

// Assumes vector size of VEC_WIDTH - must be called with img_type.VectorRange(VEC_WIDTH)
// Type must be specified when compiling this file, example : for unsigned 8 bit "-D U8"

// Optimization note : On my GTX 680 - fastest version is with no WITH_PADDING and VEC_WIDTH==8

#define VEC_WIDTH 8    // Number of items done in parralel per worker - Can be 4, 8 or 16

#ifdef S8
#define SCALAR char
#endif

#ifdef U8
#define SCALAR uchar
#endif

#ifdef S16
#define SCALAR short
#endif

#ifdef U16
#define SCALAR ushort
#endif

#ifdef S32
#define SCALAR int
#endif

#ifdef U32
#define SCALAR uint
#endif

#ifdef F32
#define SCALAR float
#define FLOAT
#endif

#ifndef SCALAR
#define SCALAR uchar
#endif

#define INPUT_SPACE global    // If input images are read only, they can be set to be in "constant" memory space, with possible speed improvements

#define CONCATENATE(a, b) _CONCATENATE(a, b)
#define _CONCATENATE(a, b) a ## b

#define TYPE CONCATENATE(SCALAR, VEC_WIDTH)  // Example : uchar16

#define CONVERT(val) CONCATENATE(CONCATENATE(convert_, TYPE), _sat) (val)           // Example : convert_uchar16_sat(val)
#define CONVERT_SCALAR(val) CONCATENATE(CONCATENATE(convert_, SCALAR), _sat) (val)  // Example : convert_uchar_sat(val)
#define CONVERT_FLOAT(val) CONCATENATE(convert_float, VEC_WIDTH) (val)              // Example : convert_float16(val)

#define BEGIN  \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src_step /= sizeof(SCALAR);\
   dst_step /= sizeof(SCALAR);

#define BEGIN2 \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src1_step /= sizeof(SCALAR);\
   src2_step /= sizeof(SCALAR);\
   dst_step /= sizeof(SCALAR);

#define WITH_PADDING

#ifdef WITH_PADDING
// This versions is capable of handling all types of image, including :
//    Width that is not a multiple of VEC_WIDTH
//    Images with step larger than Width * channels (ROI or Images with padding)
// If using a small ROI on a big image, this version will be faster

#define PREPARE_SCALAR(i) \
   const INPUT_SPACE SCALAR * src_scalar = (const INPUT_SPACE SCALAR *) source;\
   global SCALAR * dst_scalar = (global SCALAR *)dest;\
   const SCALAR src = src_scalar[(gy * src_step) + i];\
   global SCALAR * dst = dst_scalar + (gy * dst_step) + i;

#define PREPARE_SCALAR2(i) \
   const INPUT_SPACE SCALAR * src1_scalar = (const INPUT_SPACE SCALAR *) source1;\
   const INPUT_SPACE SCALAR * src2_scalar = (const INPUT_SPACE SCALAR *) source2;\
   global SCALAR * dst_scalar = (global SCALAR *) dest;\
   const SCALAR src1 = src1_scalar[(gy * src1_step) + i];\
   const SCALAR src2 = src2_scalar[(gy * src2_step) + i];\
   global SCALAR * dst = dst_scalar + (gy * dst_step) + i;

#define PREPARE_VECTOR \
   const TYPE src = source[(gy * src_step) / VEC_WIDTH + gx];\
   global TYPE * dst = dest + (gy * dst_step) / VEC_WIDTH + gx;

#define PREPARE_VECTOR2 \
   const TYPE src1 = source1[(gy * src1_step) / VEC_WIDTH + gx];\
   const TYPE src2 = source2[(gy * src2_step) / VEC_WIDTH + gx];\
   global TYPE * dst = dest + (gy * dst_step) / VEC_WIDTH + gx;

// TODO : Test performance with one worker per scalar instead of a loop
#define LAST_WORKER(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR(i)\
         *dst = code;\
      }\
      return;\
   }

#define LAST_WORKER2(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR2(i)\
         *dst = code;\
      }\
      return;\
   }

#else // WITH_PADDING

// This versions is simpler and a bit faster. It only handles images that have :
//    Width that is a multiple of VEC_WIDTH
//       Improper width will cause the last few pixels not to be processed
//    Step == Width * Channels * Depth / 8 (no ROI and no padding at the end of the line)
//       All data between the image lines will be processed
// This version needs a 1D Range : cl::NDRange(Width * Height * Channels / VEC_WIDTH, 1, 1)

#define PREPARE_VECTOR \
   const TYPE src = source[gx];\
   global TYPE * dst = dest + gx;

#define PREPARE_VECTOR2 \
   const TYPE src1 = source1[gx];\
   const TYPE src2 = source2[gx];\
   global TYPE * dst = dest + gx;

#define LAST_WORKER(code)
#define LAST_WORKER2(code)

#endif   // WITH_PADDING



#define BINARY_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source1, INPUT_SPACE const TYPE * source2,\
                global TYPE * dest, int src1_step, int src2_step, int dst_step, int width)\
{\
   BEGIN2\
   LAST_WORKER2(code)\
   PREPARE_VECTOR2\
   *dst = code;\
}

#define CONSTANT_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width, uint value_in)\
{\
   BEGIN\
   SCALAR value = CONVERT_SCALAR(value_in);\
   LAST_WORKER(code)\
   PREPARE_VECTOR\
   *dst = code;\
}

#define UNARY_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width)\
{\
   BEGIN\
   LAST_WORKER(code)\
   PREPARE_VECTOR\
   *dst = code;\
}


// Bitwise operations - float not allowed
#ifndef FLOAT

BINARY_OP(and_images, src1 & src2)
BINARY_OP(or_images, src1 | src2)
BINARY_OP(xor_images, src1 ^ src2)
// image and value
CONSTANT_OP(and_constant, src & value)
CONSTANT_OP(or_constant, src | value)
CONSTANT_OP(xor_constant, src ^ value)

// Unary
UNARY_OP(not_image, ~src)

#endif   // not FLOAT
