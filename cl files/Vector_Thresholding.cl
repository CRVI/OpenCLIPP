////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_Thresholding.cl
//! @date   : Mar 2014
//!
//! @brief  : Thresholding operations on image buffers
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
#define FTYPE CONCATENATE(float, VEC_WIDTH)  // Example : uchar16
#define TYPE_U8 CONCATENATE(uchar, VEC_WIDTH) // Example : uchar16

#ifndef FLOAT
#define CONVERT(val) CONCATENATE(CONCATENATE(convert_, TYPE), _sat) (val)           // Example : convert_uchar16_sat(val)
#define CONVERT_SCALAR(val) CONCATENATE(CONCATENATE(convert_, SCALAR), _sat) (val)  // Example : convert_uchar_sat(val)
#define CONVERT_FLOAT(val) CONCATENATE(convert_float, VEC_WIDTH) (val)              // Example : convert_float16(val)
#else
#define CONVERT(val) val
#define CONVERT_SCALAR(val) val
#define CONVERT_FLOAT(val) val
#endif

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

#define BEGIN_U8  \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src_step /= sizeof(SCALAR);\
   dst_step /= sizeof(uchar);

#define BEGIN2_U8 \
   const int gx = get_global_id(0);	/* x divided by VEC_WIDTH */ \
   const int gy = get_global_id(1);\
   src1_step /= sizeof(SCALAR);\
   src2_step /= sizeof(SCALAR);\
   dst_step /= sizeof(uchar);


#define WITH_PADDING

#ifdef WITH_PADDING
// This versions is capable of handling all types of image, including :
//    Width that is not a multiple of VEC_WIDTH
//    Images with step larger than Width * channels (ROI or Images with padding)
// If using a small ROI on a big image, this version will be faster

	#define PREPARE_SCALAR_U8(i) \
	   const INPUT_SPACE SCALAR * src_scalar = (const INPUT_SPACE SCALAR *) source;\
	   global uchar * dst_scalar = (global uchar *)dest;\
	   const float src = convert_float(src_scalar[(gy * src_step) + i]);

	#define PREPARE_SCALAR(i) \
	   const INPUT_SPACE SCALAR * src_scalar = (const INPUT_SPACE SCALAR *) source;\
	   global SCALAR * dst_scalar = (global SCALAR *)dest;\
	   const float src = convert_float(src_scalar[(gy * src_step) + i]);


	#define PREPARE_SCALAR2_U8(i) \
	   const INPUT_SPACE SCALAR * src1_scalar = (const INPUT_SPACE SCALAR *) source1;\
	   const INPUT_SPACE SCALAR * src2_scalar = (const INPUT_SPACE SCALAR *) source2;\
	   global uchar * dst_scalar = (global uchar *)dest;\
	   const float src1 = convert_float(src1_scalar[(gy * src1_step) + i]);\
	   const float src2 = convert_float(src2_scalar[(gy * src2_step) + i]);

	#define PREPARE_SCALAR2(i) \
	   const INPUT_SPACE SCALAR * src1_scalar = (const INPUT_SPACE SCALAR *) source1;\
	   const INPUT_SPACE SCALAR * src2_scalar = (const INPUT_SPACE SCALAR *) source2;\
	   global SCALAR * dst_scalar = (global SCALAR *)dest;\
	   const float src1 = convert_float(src1_scalar[(gy * src1_step) + i]);\
	   const float src2 = convert_float(src2_scalar[(gy * src2_step) + i]);


#define SCALAR_OP_U8(code) dst_scalar[(gy * dst_step) + i] = convert_uchar_sat(code.s0)
#define SCALAR_OP(code) dst_scalar[(gy * dst_step) + i] = CONVERT_SCALAR(code)

#define PREPARE_VECTOR \
   const FTYPE src = CONVERT_FLOAT(source[(gy * src_step) / VEC_WIDTH + gx]);

#define PREPARE_VECTOR2 \
   const FTYPE src1 = CONVERT_FLOAT(source1[(gy * src1_step) / VEC_WIDTH + gx]);\
   const FTYPE src2 = CONVERT_FLOAT(source2[(gy * src2_step) / VEC_WIDTH + gx]);

#define VECTOR_OP_U8(code) dest[(gy * dst_step) / VEC_WIDTH + gx] = CONCATENATE(CONCATENATE(convert_, TYPE_U8), _sat)(code)
#define VECTOR_OP(code) dest[(gy * dst_step) / VEC_WIDTH + gx] = CONVERT(code)

// TODO : Test performance with one worker per scalar instead of a loop
#define LAST_WORKER(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR(i)\
         SCALAR_OP(code);\
      }\
      return;\
   }

#define LAST_WORKER_U8(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR_U8(i)\
         SCALAR_OP_U8(code);\
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
         SCALAR_OP(code);\
      }\
      return;\
   }

#define LAST_WORKER2_U8(code) \
   if ((gx + 1) * VEC_WIDTH > width)\
   {\
      /* Last worker on the current row for an image that has a width that is not a multiple of VEC_WIDTH*/\
      for (int i = gx * VEC_WIDTH; i < width; i++)\
      {\
         PREPARE_SCALAR2_U8(i)\
         SCALAR_OP_U8(code);\
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
   const FTYPE src = CONVERT_FLOAT(source[gx]);

#define PREPARE_VECTOR2 \
   const FTYPE src1 = CONVERT_FLOAT(source1[gx]);\
   const FTYPE src2 = CONVERT_FLOAT(source2[gx]);

#define VECTOR_OP_U8(code) dest[gx] = CONCATENATE(CONCATENATE(convert_, TYPE_U8), _sat)(code)
#define VECTOR_OP(code) dest[gx] = CONVERT(code)

#define LAST_WORKER(code)
#define LAST_WORKER2(code)
#define LAST_WORKER_U8(code)
#define LAST_WORKER2_U8(code)

#endif   // WITH_PADDING

kernel void thresholdLT(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width, float thresh, float valueLower)
{
   BEGIN\
   LAST_WORKER(src < thresh ? valueLower : src)\
   PREPARE_VECTOR\
   VECTOR_OP(src < thresh ? valueLower : src);\
}

kernel void thresholdGT(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width, float thresh, float valueHigher)
{
   BEGIN\
   LAST_WORKER(src > thresh ? valueHigher : src)\
   PREPARE_VECTOR\
   VECTOR_OP(src > thresh ? valueHigher : src);\
}

kernel void thresholdGTLT(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width, 
						  float threshLT, float valueLower, float threshGT, float valueHigher)
{
   BEGIN\
   LAST_WORKER(src < threshLT ? valueLower : src > threshGT ? valueHigher : src)\
   PREPARE_VECTOR\
   VECTOR_OP(src < threshLT ? valueLower : src > threshGT ? valueHigher : src);\
}

#define BINARY_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source1, INPUT_SPACE const TYPE * source2,\
                global TYPE * dest, int src1_step, int src2_step, int dst_step, int width)\
{\
   BEGIN2\
   LAST_WORKER2(code)\
   PREPARE_VECTOR2\
   VECTOR_OP(code);\
}

/*#define CONSTANT_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source, global TYPE * dest, int src_step, int dst_step, int width, float value)\
{\
   BEGIN\
   LAST_WORKER(code)\
   PREPARE_VECTOR\
   VECTOR_OP(code);\
}*/

BINARY_OP(img_thresh_LT, (src1 < src2 ? src1 : src2))
BINARY_OP(img_thresh_LQ, (src1 <= src2 ? src1 : src2))
BINARY_OP(img_thresh_EQ, (src1 == src2 ? src1 : src2))
BINARY_OP(img_thresh_GQ, (src1 >= src2 ? src1 : src2))
BINARY_OP(img_thresh_GT, (src1 > src2 ? src1 : src2))

//-------------------------------------------------------------------------------------------------



#define WHITE ((CONCATENATE(uint, VEC_WIDTH))(255))
#define BLACK ((CONCATENATE(uint, VEC_WIDTH))(0))

//#undef VECTOR_OP

//#define VECTOR_OP(code) dest[gx] = CONCATENATE(CONCATENATE(convert_, TYPE_U8), _sat)(code)

//-------------------------------------------------------------------------------------------------

#undef BINARY_OP

#define BINARY_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source1, INPUT_SPACE const TYPE * source2,\
                global TYPE_U8 * dest, int src1_step, int src2_step, int dst_step, int width)\
{\
   BEGIN2_U8\
   LAST_WORKER2_U8(code)\
   PREPARE_VECTOR2\
   VECTOR_OP_U8(code);\
}

#define CONSTANT_OP(name, code) \
__attribute__(( vec_type_hint(TYPE) ))\
kernel void name(INPUT_SPACE const TYPE * source, global TYPE_U8 * dest, int src_step, int dst_step, int width, float value)\
{\
   BEGIN_U8\
   LAST_WORKER_U8(code)\
   PREPARE_VECTOR\
   VECTOR_OP_U8(code);\
}

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

