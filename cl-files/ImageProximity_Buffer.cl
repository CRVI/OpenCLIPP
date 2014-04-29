////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_ImageProximity.cl
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on image buffers
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

#define INPUT     global const TYPE *
#define OUTPUT    global DEST_TYPE *

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   src_step  /= sizeof(TYPE);\
   temp_step /= sizeof(TYPE);\
   dst_step  /= sizeof(DEST_TYPE);

#define CONVERT_INTERNAL(val) CONCATENATE(convert_, INTERNAL) (val)              // Example : convert_float4(val)

#ifndef FLOAT
#define CONVERT(v) CONCATENATE(convert_, DEST_TYPE) (v)
#else    // FLOAT
#define CONVERT(v) (v)
#endif   // FLOAT


#define READ(img, step, x, y)  CONVERT_INTERNAL(img[y * step + x])
#define WRITE(img, val) img[get_global_id(1) * dst_step + get_global_id(0)] = CONVERT(val)

#define DIM_ARGS   int src_step, int temp_step, int dst_step, int templateWidth, int templateHeight, int dest_width, int dest_height

// 1 Channel 
#define TYPE      SCALAR
#define DEST_TYPE float
#define SUFFIX    _1C
#define INTERNAL  float
#include "ImageProximity.Impl.h"

// 2 Channels
#undef  TYPE
#undef  SUFFIX
#undef  INTERNAL
#define TYPE      TYPE2
#define DEST_TYPE float2
#define SUFFIX    _2C
#define INTERNAL  float2
#include "ImageProximity.Impl.h"

// 3 Channels 
#undef  TYPE
#undef  SUFFIX
#undef  INTERNAL
#define TYPE      TYPE3
#define DEST_TYPE float3
#define SUFFIX    _3C
#define INTERNAL  float3
#include "ImageProximity.Impl.h"

// 4 Channels
#undef  TYPE
#undef  SUFFIX
#undef  INTERNAL
#define TYPE      TYPE4
#define DEST_TYPE float4
#define SUFFIX    _4C
#define INTERNAL  float4
#include "ImageProximity.Impl.h"
