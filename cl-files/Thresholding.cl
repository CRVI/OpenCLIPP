////////////////////////////////////////////////////////////////////////////////
//! @file	: Thresholding.cl
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

#include "Images.h"


kernel void thresholdGTLT(INPUT source, OUTPUT dest, float threshLT,
                         float valueLower, float threshGT, float valueHigher)
{
   BEGIN

   // Read pixel
   TYPE src_color = READ_IMAGE(source, pos);

   // Modify color
   TYPE dst_color = (src_color < (TYPE)(threshLT) ? (TYPE)(valueLower) : src_color);
   dst_color = (src_color > (TYPE)(threshGT) ? (TYPE)(valueHigher) : dst_color);

   // Write pixel
   WRITE_IMAGE(dest, pos, dst_color);
}



#define DST_TYPE TYPE

#define BINARY_OP(name, code)\
kernel void name(INPUT source1, INPUT source2, OUTPUT dest)\
{\
   BEGIN\
   TYPE src1 = READ_IMAGE(source1, pos);\
   TYPE src2 = READ_IMAGE(source2, pos);\
   DST_TYPE dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}

#define CONSTANT_OP(name, code)\
kernel void name(INPUT source, OUTPUT dest, float value)\
{\
   BEGIN\
   TYPE src = READ_IMAGE(source, pos);\
   DST_TYPE dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
}

#define THRESHOLD_OP(name, code)\
kernel void name(INPUT source, OUTPUT dest, float thresh_arg, float value_arg)\
{\
   BEGIN\
   SCALAR thresh = thresh_arg;\
   SCALAR value = value_arg;\
   TYPE src = READ_IMAGE(source, pos);\
   DST_TYPE dst = code;\
   WRITE_IMAGE(dest, pos, dst);\
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


// The following kernels will receive an unsigned integer (U8) image for destination
#define WHITE ((uint4)(255, 255, 255, 0))
#define BLACK ((uint4)(0, 0, 0, 0))

#undef WRITE_IMAGE
#define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)

#undef DST_TYPE
#define DST_TYPE uint4

BINARY_OP(img_compare_LT, (src1 <  src2 ? WHITE : BLACK))
BINARY_OP(img_compare_LQ, (src1 <= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_EQ, (src1 == src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GQ, (src1 >= src2 ? WHITE : BLACK))
BINARY_OP(img_compare_GT, (src1 >  src2 ? WHITE : BLACK))

CONSTANT_OP(compare_LT, (src <  (TYPE)(value) ? WHITE : BLACK))
CONSTANT_OP(compare_LQ, (src <= (TYPE)(value) ? WHITE : BLACK))
CONSTANT_OP(compare_EQ, (src == (TYPE)(value) ? WHITE : BLACK))
CONSTANT_OP(compare_GQ, (src >= (TYPE)(value) ? WHITE : BLACK))
CONSTANT_OP(compare_GT, (src >  (TYPE)(value) ? WHITE : BLACK))
