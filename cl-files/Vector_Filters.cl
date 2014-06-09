////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_Filters.cl
//! @date   : Jan 2014
//!
//! @brief  : Convolution-style filters on image buffers
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

#define INPUT     INPUT_SPACE const TYPE *
#define OUTPUT    global TYPE *
#define INTERNAL  REAL

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };\
   src_step /= sizeof(TYPE);

#define CONVERT_INTERNAL(val) CONCATENATE(convert_, INTERNAL) (val)              // Example : convert_float4(val)


#define READ(img, pos)  CONVERT_INTERNAL(img[(pos).y * src_step + (pos).x])
#define WRITE(img, val) img[get_global_id(1) * dst_step / sizeof(TYPE) + get_global_id(0)] = CONVERT(val)


#define DIM_ARGS    , int src_step, int dst_step, int width, int height
#define DIMS        , src_step, dst_step, width, height

bool OutsideImage(int2 pos, int src_step, int dst_step, int width, int height, int mask_size)
{
   if (pos.x < mask_size || pos.y < mask_size)
      return true;

   if (pos.x >= width - mask_size || pos.y >= height - mask_size)
      return true;

   return false;
}


// Include macros for median filters
#include "Median.h"

#define LW 16  // Cached version of median filter uses a 16x16 local size


// Actual code for filter operations is in the file Filters.Impl.h

#include "Filters.Impl.h"
