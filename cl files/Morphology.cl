////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.cl
//! @date   : Jul 2013
//!
//! @brief  : Morphological operations on images
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


constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


#ifdef I

   // For signed integer images
   #define READ_IMAGE(img, pos) read_imagei(img, sampler, pos)
   #define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, px)
   #define TYPE int4

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, px)
      #define TYPE uint4

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) read_imagef(img, sampler, pos)
      #define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, px)
      #define TYPE float4
      #define FLOAT

   #endif // UI

#endif // I


#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#define CONCATENATE(a, b) _CONCATENATE(a, b)
#define _CONCATENATE(a, b) a ## b


#define ERODE(color, new_pixel) min(color, new_pixel)
#define DILATE(color, new_pixel) max(color, new_pixel)

#define NEIGHBOUR_CASE(size, operation) \
   case size:\
   {\
      for (int x = -size; x <= size; x++)\
         for (int y = -size; y <= size; y++)\
         {\
            const int2 pos2 = { gx + x, gy + y};\
            color = operation(color, READ_IMAGE(source, pos2));\
         }\
   }\
   break;

#define NEIGHBOUR_KERNEL(name, operation) \
kernel void name(read_only image2d_t source, write_only image2d_t dest, int width)\
{\
   BEGIN\
   TYPE color = READ_IMAGE(source, pos);\
   private const int mask_size = width / 2;\
   switch(mask_size)\
   {\
   NEIGHBOUR_CASE(1, operation) /* 3x3 */\
   NEIGHBOUR_CASE(2, operation) /* 5x5 */\
   NEIGHBOUR_CASE(3, operation) /* 7x7 */\
   NEIGHBOUR_CASE(4, operation) /* 9x9 */\
   NEIGHBOUR_CASE(5, operation) /* 11x11 */\
   NEIGHBOUR_CASE(6, operation) /* 13x13 */\
   NEIGHBOUR_CASE(7, operation) /* 15x */\
   NEIGHBOUR_CASE(8, operation) /* 17x */\
   NEIGHBOUR_CASE(9, operation) /* 19x */\
   NEIGHBOUR_CASE(10, operation) /* 21x */\
   NEIGHBOUR_CASE(11, operation) /* 23x */\
   NEIGHBOUR_CASE(12, operation) /* 25x */\
   NEIGHBOUR_CASE(13, operation) /* 27x */\
   NEIGHBOUR_CASE(14, operation) /* 29x */\
   NEIGHBOUR_CASE(15, operation) /* 31x */\
   NEIGHBOUR_CASE(16, operation) /* 33x */\
   NEIGHBOUR_CASE(17, operation) /* 35x */\
   NEIGHBOUR_CASE(18, operation) /* 37x */\
   NEIGHBOUR_CASE(19, operation) /* 39x */\
   NEIGHBOUR_CASE(20, operation) /* 41x */\
   NEIGHBOUR_CASE(21, operation) /* 43x */\
   NEIGHBOUR_CASE(22, operation) /* 45x */\
   NEIGHBOUR_CASE(23, operation) /* 47x */\
   NEIGHBOUR_CASE(24, operation) /* 49x */\
   NEIGHBOUR_CASE(25, operation) /* 51x */\
   NEIGHBOUR_CASE(26, operation) /* 53x */\
   NEIGHBOUR_CASE(27, operation) /* 55x */\
   NEIGHBOUR_CASE(28, operation) /* 57x */\
   NEIGHBOUR_CASE(29, operation) /* 59x */\
   NEIGHBOUR_CASE(30, operation) /* 61x */\
   NEIGHBOUR_CASE(31, operation) /* 63x */\
   }\
   WRITE_IMAGE(dest, pos, color);\
}\
kernel void CONCATENATE(name, 3) (read_only image2d_t source, write_only image2d_t dest)\
{\
   BEGIN\
   TYPE color = READ_IMAGE(source,             (int2)(gx - 1, gy - 1));\
   color = operation(color, READ_IMAGE(source, (int2)(gx    , gy - 1)));\
   color = operation(color, READ_IMAGE(source, (int2)(gx + 1, gy - 1)));\
   color = operation(color, READ_IMAGE(source, (int2)(gx - 1, gy    )));\
   color = operation(color, READ_IMAGE(source, (int2)(gx    , gy    )));\
   color = operation(color, READ_IMAGE(source, (int2)(gx + 1, gy    )));\
   color = operation(color, READ_IMAGE(source, (int2)(gx - 1, gy + 1)));\
   color = operation(color, READ_IMAGE(source, (int2)(gx    , gy + 1)));\
   color = operation(color, READ_IMAGE(source, (int2)(gx + 1, gy + 1)));\
   WRITE_IMAGE(dest, pos, color);\
}

NEIGHBOUR_KERNEL(erode, ERODE)
NEIGHBOUR_KERNEL(dilate, DILATE)


kernel void sub_images(read_only image2d_t source1, read_only image2d_t source2, write_only image2d_t dest)
{
   BEGIN
   TYPE src1 = READ_IMAGE(source1, pos);
   TYPE src2 = READ_IMAGE(source2, pos);
   WRITE_IMAGE(dest, pos, src1 - src2);
}
