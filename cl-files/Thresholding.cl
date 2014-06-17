////////////////////////////////////////////////////////////////////////////////
//! @file	: Thresholding.cl
//! @date   : Mar 2014
//!
//! @brief  : Thresholding operations
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

#include "Vector.h"


kernel void thresholdGTLT(INPUT_SPACE const SCALAR * source, global DST_SCALAR * dest, int src_step, int dst_step, int width, 
                    float threshLT, float valueLower, float threshGT, float valueHigher)
{
   BEGIN
   SCALAR_CODE(src < threshLT ? valueLower : (src > threshGT ? valueHigher : src))
   PREPARE_VECTOR
   VECTOR_OP(src < (TYPE)((SCALAR)threshLT) ? (DST)((DST_SCALAR)valueLower) : (src > (TYPE)((SCALAR)threshGT) ? (DST)((DST_SCALAR)valueHigher) : src));
}


#define THRESHOLD_OP(name, code) \
kernel void name(INPUT_SPACE const SCALAR * source, global DST_SCALAR * dest, int src_step, int dst_step, int width, float thresh_arg, float value_arg)\
{\
   BEGIN\
   INTERNAL_SCALAR value = value_arg;\
   INTERNAL_SCALAR thresh = thresh_arg;\
   SCALAR_CODE(code)\
   PREPARE_VECTOR\
   VECTOR_OP(code);\
}

THRESHOLD_OP(threshold_LT, (src <  thresh ? value : src))
THRESHOLD_OP(threshold_LQ, (src <= thresh ? value : src))
THRESHOLD_OP(threshold_EQ, (src == thresh ? value : src))
THRESHOLD_OP(threshold_GQ, (src >= thresh ? value : src))
THRESHOLD_OP(threshold_GT, (src >  thresh ? value : src))


BINARY_OP(img_thresh_LT, (src1 <  src2 ? src1 : src2))
BINARY_OP(img_thresh_LQ, (src1 <= src2 ? src1 : src2))
BINARY_OP(img_thresh_EQ, (src1 == src2 ? src1 : src2))
BINARY_OP(img_thresh_GQ, (src1 >= src2 ? src1 : src2))
BINARY_OP(img_thresh_GT, (src1 >  src2 ? src1 : src2))



// The following kernels will receive an unsigned char (U8) image for destination
#undef DST_SCALAR
#undef DST
#define DST_SCALAR uchar
#define DST CONCATENATE(DST_SCALAR, VEC_WIDTH)

#define TST_TYPE CONCATENATE(uint, VEC_WIDTH)

#undef CONVERT_DST
#undef CONVERT_DST_SCALAR
#define CONVERT_DST(val) CONCATENATE(CONCATENATE(convert_, DST), _sat) (val)
#define CONVERT_DST_SCALAR(val) (val)

#define WHITE ((INTERNAL)(255))
#define BLACK ((INTERNAL)(0))

#ifdef SCALAR_OP
#undef SCALAR_OP
#endif
#define SCALAR_OP(code) WRITE_SCALAR(dest, dst_step, i, gy, (code).x)


BINARY_OP(img_compare_LT, (src1 <  src2 ? WHITE : BLACK))
BINARY_OP(img_compare_LQ, (src1 <= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_EQ, (src1 == src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GQ, (src1 >= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GT, (src1 >  src2 ? WHITE : BLACK))

CONSTANT_OP(compare_LT, (src <  value ? WHITE : BLACK))
CONSTANT_OP(compare_LQ, (src <= value ? WHITE : BLACK))
CONSTANT_OP(compare_EQ, (src == value ? WHITE : BLACK))
CONSTANT_OP(compare_GQ, (src >= value ? WHITE : BLACK))
CONSTANT_OP(compare_GT, (src >  value ? WHITE : BLACK))
