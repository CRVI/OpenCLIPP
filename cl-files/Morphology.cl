////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.cl
//! @date   : Jan 2014
//!
//! @brief  : Morphological operations
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

#ifndef LW
#define LW 16  // local width - kernels using local cache need to use a local range of LWxLW
#endif


#define BEGIN  \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   src_step /= sizeof(TYPE);\
   dst_step /= sizeof(TYPE);

#define INPUT  INPUT_SPACE const TYPE *
#define OUTPUT global TYPE *

#define MORPHOLOGY_IMPL(name, op, mask_width) \
kernel void CONCATENATE(name, mask_width) (INPUT source, OUTPUT dest, int src_step, int dst_step, int width, int height)\
{\
   BEGIN\
   \
   TYPE Value = source[gy * src_step + gx];\
   \
   const int mask_size = mask_width / 2;\
   \
   if (gy - mask_size < 0 || gy + mask_size >= height || gx - mask_size < 0 || gx + mask_size >= width)\
   {\
      /* Would look outside of image - Save unmodified result*/\
      dest[gy * dst_step + gx] = Value;\
      return;\
   }\
   \
   for (int y = -mask_size; y <= mask_size; y++)\
   {\
      int py = gy + y;\
      for (int x = -mask_size; x <= mask_size; x++)\
      {\
         TYPE Val = source[py * src_step + gx + x];\
         Value = op(Val, Value);\
      }\
      \
   }\
   \
   /* Save result */\
   dest[gy * dst_step + gx] = Value;\
}


#define MORPHOLOGY_CACHED(name, op, mask_width) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void CONCATENATE(CONCATENATE(name, mask_width), _cached) (INPUT source, OUTPUT dest, int src_step, int dst_step, int width, int height)\
{\
   BEGIN\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   \
   TYPE Value = source[gy * src_step + gx];\
   \
   /* Cache pixels */\
   local TYPE cache[LW * LW];\
   cache[lid] = Value;\
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   /* Cache information */\
   int bitmask = LW - 1;\
   int x_cache_begin = gx - (gx & bitmask);\
   int x_cache_end = x_cache_begin + LW;\
   int y_cache_begin = gy - (gy & bitmask);\
   int y_cache_end = y_cache_begin + LW;\
   \
   const int mask_size = mask_width / 2;\
   \
   if (gy - mask_size < 0 || gy + mask_size >= height || gx - mask_size < 0 || gx + mask_size >= width)\
   {\
      /* Would look outside of image - Save unmodified result */\
      dest[gy * dst_step + gx] = Value;\
      return;\
   }\
   \
   for (int y = -mask_size; y <= mask_size; y++)\
   {\
      int py = gy + y;\
      \
      if (py < y_cache_begin || py >= y_cache_end)\
      {\
         for (int x = -mask_size; x <= mask_size; x++)\
         {\
            TYPE Val = source[py * src_step + gx + x];\
            Value = op(Val, Value);\
         }\
      \
      }\
      else\
      {\
         for (int x = -mask_size; x <= mask_size; x++)\
         {\
            int px = gx + x;\
            TYPE Val;\
            if (px < x_cache_begin || px >= x_cache_end)\
               Val = source[py * src_step + px];\
            else\
            {\
               /* Read from cache */\
               int cache_y = py - y_cache_begin;\
               int cache_x = px - x_cache_begin;\
               Val = cache[cache_y * LW + cache_x];\
            }\
            \
            Value = op(Val, Value);\
         }\
         \
      }\
      \
   }\
   \
   /* Save result */\
   dest[gy * dst_step + gx] = Value;\
}

// Each size has a standard version and a version with local cache - host code will choose which version to use
#define MORPHOLOGY(name, op) \
   MORPHOLOGY_IMPL(name, op, 3) \
   MORPHOLOGY_IMPL(name, op, 5) \
   MORPHOLOGY_IMPL(name, op, 7) \
   MORPHOLOGY_IMPL(name, op, 9) \
   MORPHOLOGY_IMPL(name, op, 11) \
   MORPHOLOGY_IMPL(name, op, 13) \
   MORPHOLOGY_IMPL(name, op, 15) \
   MORPHOLOGY_IMPL(name, op, 17) \
   MORPHOLOGY_IMPL(name, op, 19) \
   MORPHOLOGY_IMPL(name, op, 21) \
   MORPHOLOGY_IMPL(name, op, 23) \
   MORPHOLOGY_IMPL(name, op, 25) \
   MORPHOLOGY_IMPL(name, op, 27) \
   MORPHOLOGY_IMPL(name, op, 29) \
   MORPHOLOGY_IMPL(name, op, 31) \
   MORPHOLOGY_IMPL(name, op, 33) \
   MORPHOLOGY_IMPL(name, op, 35) \
   MORPHOLOGY_IMPL(name, op, 37) \
   MORPHOLOGY_IMPL(name, op, 39) \
   MORPHOLOGY_IMPL(name, op, 41) \
   MORPHOLOGY_IMPL(name, op, 43) \
   MORPHOLOGY_IMPL(name, op, 45) \
   MORPHOLOGY_IMPL(name, op, 47) \
   MORPHOLOGY_IMPL(name, op, 49) \
   MORPHOLOGY_IMPL(name, op, 51) \
   MORPHOLOGY_IMPL(name, op, 53) \
   MORPHOLOGY_IMPL(name, op, 55) \
   MORPHOLOGY_IMPL(name, op, 57) \
   MORPHOLOGY_IMPL(name, op, 59) \
   MORPHOLOGY_IMPL(name, op, 61) \
   MORPHOLOGY_IMPL(name, op, 63) \
   MORPHOLOGY_CACHED(name, op, 3) \
   MORPHOLOGY_CACHED(name, op, 5) \
   MORPHOLOGY_CACHED(name, op, 7) \
   MORPHOLOGY_CACHED(name, op, 9) \
   MORPHOLOGY_CACHED(name, op, 11) \
   MORPHOLOGY_CACHED(name, op, 13) \
   MORPHOLOGY_CACHED(name, op, 15) \
   MORPHOLOGY_CACHED(name, op, 17) \
   MORPHOLOGY_CACHED(name, op, 19) \
   MORPHOLOGY_CACHED(name, op, 21) \
   MORPHOLOGY_CACHED(name, op, 23) \
   MORPHOLOGY_CACHED(name, op, 25) \
   MORPHOLOGY_CACHED(name, op, 27) \
   MORPHOLOGY_CACHED(name, op, 29) \
   MORPHOLOGY_CACHED(name, op, 31) \
   MORPHOLOGY_CACHED(name, op, 33) \
   MORPHOLOGY_CACHED(name, op, 35) \
   MORPHOLOGY_CACHED(name, op, 37) \
   MORPHOLOGY_CACHED(name, op, 39) \
   MORPHOLOGY_CACHED(name, op, 41) \
   MORPHOLOGY_CACHED(name, op, 43) \
   MORPHOLOGY_CACHED(name, op, 45) \
   MORPHOLOGY_CACHED(name, op, 47) \
   MORPHOLOGY_CACHED(name, op, 49) \
   MORPHOLOGY_CACHED(name, op, 51) \
   MORPHOLOGY_CACHED(name, op, 53) \
   MORPHOLOGY_CACHED(name, op, 55) \
   MORPHOLOGY_CACHED(name, op, 57) \
   MORPHOLOGY_CACHED(name, op, 59) \
   MORPHOLOGY_CACHED(name, op, 61) \
   MORPHOLOGY_CACHED(name, op, 63)

MORPHOLOGY(erode, min)
MORPHOLOGY(dilate, max)
