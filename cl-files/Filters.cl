////////////////////////////////////////////////////////////////////////////////
//! @file	: Filters.cl
//! @date   : Jul 2013
//!
//! @brief  : Convolution-style filters on images
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

#define CONVERT_INTERNAL(val) CONCATENATE(convert_, INTERNAL) (val)              // Example : convert_float4(val)

#define DIM_ARGS
#define DIMS


#define READ(img, pos)   CONVERT_INTERNAL(READ_IMAGE(img, pos))
#define WRITE(img, val)  WRITE_IMAGE(img, (int2)(get_global_id(0), get_global_id(1)), CONVERT(val))


bool OutsideImage(int2 pos, int mask_size)
{
   // We can read safely outside of the image so we ignore outside of image situations
   return false;
}


// Include macros for median filters
#include "Median.h"

#define LW 16  // Cached version of median filter uses a 16x16 local size


// Actual code for filter operations is in the file Filters.Impl.h

// 1 Channel 
#define SUFFIX    _1C
#define INTERNAL  float
#define READ(img, pos)   CONVERT_INTERNAL(READ_IMAGE(img, pos).x)
#include "Filters.Impl.h"

// 2 Channel images are currently not supported - they will be processed using the _4C kernels

// 3 Channel images do not exist in OpenCL

// 4 Channels
#undef  SUFFIX
#undef  INTERNAL
#undef  READ
#define SUFFIX    _4C
#define INTERNAL  float4
#define READ(img, pos)   CONVERT_INTERNAL(READ_IMAGE(img, pos))
#include "Filters.Impl.h"
