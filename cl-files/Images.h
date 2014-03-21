////////////////////////////////////////////////////////////////////////////////
//! @file	: Images.h
//! @date   : Mar 2014
//!
//! @brief  : Macros for manipulating image2d_t
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

#include "Base.h"

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define SAMPLER sampler

#ifdef I

   // For signed integer images
   #define READ_IMAGE(img, pos) read_imagei(img, SAMPLER, pos)
   #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, px)
   #define TYPE int4
   #define SCALAR int

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, SAMPLER, pos)
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)
      #define TYPE uint4
      #define SCALAR uint

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) read_imagef(img, SAMPLER, pos)
      #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, px)
      #define TYPE float4
      #define SCALAR float
      #define FLOAT

   #endif // UI

#endif // I


#ifdef FLOAT
#define CONVERT_FLOAT(val) val
#define ABS fabs
#define CONVERT(val) CONCATENATE(convert_, TYPE) (val)           // convert_float(val)
#else
#define CONVERT_FLOAT(val) convert_float4(val)
#define ABS abs
#define CONVERT(val) CONCATENATE(CONCATENATE(convert_, TYPE), _sat) (val)           // Example : convert_int4_sat(val)
#endif

#define TYPE2 CONCATENATE(SCALAR, 2)
#define TYPE3 CONCATENATE(SCALAR, 3)
#define TYPE4 CONCATENATE(SCALAR, 4)


#define INPUT  read_only  image2d_t
#define OUTPUT write_only image2d_t

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };
