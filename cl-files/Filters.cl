////////////////////////////////////////////////////////////////////////////////
//! @file	: Filters.cl
//! @date   : Jan 2014
//!
//! @brief  : Convolution-style filters
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

// Include macros for median filters
#include "Median.h"

#define INPUT     INPUT_SPACE const TYPE *
#define OUTPUT    global TYPE *

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };\
   src_step /= sizeof(TYPE);


#define READ(img, pos)  CONVERT_REAL(img[(pos).y * src_step + (pos).x])
#define WRITE(img, val) img[get_global_id(1) * dst_step / sizeof(TYPE) + get_global_id(0)] = CONVERT(val)


// Dimension arguments - these two macros are to reduce the size of argument lists to improve readability
#define DIM_ARGS    , int src_step, int dst_step, int width, int height
#define DIMS        , src_step, dst_step, width, height


// Serves as bounds check
bool OutsideImage(int2 pos, int src_step, int dst_step, int width, int height, int mask_size)
{
   if (pos.x < mask_size || pos.y < mask_size)
      return true;

   if (pos.x >= width - mask_size || pos.y >= height - mask_size)
      return true;

   return false;
}



#define CONVOLUTION_SWITCH\
   switch(mask_size)\
   {\
   CONVOLUTION_CASE(1) /* 3x3*/\
   CONVOLUTION_CASE(2) /* 5x5*/\
   CONVOLUTION_CASE(3) /* 7x7*/\
   CONVOLUTION_CASE(4) /* 9x9*/\
   CONVOLUTION_CASE(5) /* 11x11*/\
   CONVOLUTION_CASE(6) /* 13x13*/\
   CONVOLUTION_CASE(7) /* 15*/\
   CONVOLUTION_CASE(8) /* 17*/\
   CONVOLUTION_CASE(9) /* 19*/\
   CONVOLUTION_CASE(10) /* 21*/\
   CONVOLUTION_CASE(11) /* 23*/\
   CONVOLUTION_CASE(12) /* 25*/\
   CONVOLUTION_CASE(13) /* 27*/\
   CONVOLUTION_CASE(14) /* 29*/\
   CONVOLUTION_CASE(15) /* 31*/\
   CONVOLUTION_CASE(16) /* 33*/\
   CONVOLUTION_CASE(17) /* 35*/\
   CONVOLUTION_CASE(18) /* 37*/\
   CONVOLUTION_CASE(19) /* 39*/\
   CONVOLUTION_CASE(20) /* 41*/\
   CONVOLUTION_CASE(21) /* 43*/\
   CONVOLUTION_CASE(22) /* 45*/\
   CONVOLUTION_CASE(23) /* 47*/\
   CONVOLUTION_CASE(24) /* 49*/\
   CONVOLUTION_CASE(25) /* 51*/\
   CONVOLUTION_CASE(26) /* 53*/\
   CONVOLUTION_CASE(27) /* 55*/\
   CONVOLUTION_CASE(28) /* 57*/\
   CONVOLUTION_CASE(29) /* 59*/\
   CONVOLUTION_CASE(30) /* 61*/\
   CONVOLUTION_CASE(31) /* 63*/\
   }

#define CONVOLUTION_CASE(size)\
   case size:\
   {\
      for (int y = -size; y <= size; y++)\
         for (int x = -size; x <= size; x++)\
            sum += matrix[Index++] * READ(source, pos + (int2)(x, y));\
   }\
   break;

REAL Combine(REAL color1, REAL color2)
{
   return sqrt(color1 * color1 + color2 * color2);
}

REAL Convolution(INPUT source DIM_ARGS, CONST_ARG float * matrix, private int matrix_width)
{
   BEGIN

   const int mask_size = matrix_width / 2;

   if (OutsideImage(pos DIMS, mask_size))
      return 0;

   REAL sum = 0;
   int Index = 0;

   CONVOLUTION_SWITCH

   return sum;
}

void convolution(INPUT source, OUTPUT dest DIM_ARGS,
                    CONST_ARG float * matrix, private int matrix_width)
{
   REAL sum = Convolution(source DIMS, matrix, matrix_width);

   WRITE(dest, sum);
}

kernel void gaussian_blur(INPUT source, OUTPUT dest DIM_ARGS, constant const float * matrix, private int mask_size)
{
   // Does gaussian blur on first channel of image - receives a pre-calculated mask

   BEGIN

   if (OutsideImage(pos DIMS, mask_size))
      return;

   private REAL sum = 0;
   int Index = 0;

   CONVOLUTION_SWITCH

   WRITE(dest, sum);
}

kernel void gaussian3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      1.f/16, 2.f/16, 1.f/16,
      2.f/16, 4.f/16, 2.f/16,
      1.f/16, 2.f/16, 1.f/16};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void gaussian5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
       2.f/571,  7.f/571,  12.f/571,  7.f/571,  2.f/571,
       7.f/571, 31.f/571,  52.f/571, 31.f/571,  7.f/571,
      12.f/571, 52.f/571, 127.f/571, 52.f/571, 12.f/571,
       7.f/571, 31.f/571,  52.f/571, 31.f/571,  7.f/571,
       2.f/571,  7.f/571,  12.f/571,  7.f/571,  2.f/571};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void sobelH3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1, -2, -1,
       0,  0,  0,
       1,  2,  1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void sobelV3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      1, 0, -1,
      2, 0, -2,
      1, 0, -1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void sobelH5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
      -1, -4,  -6, -4, -1,
      -2, -8, -12, -8, -2,
       0,  0,   0,  0,  0,
       2,  8,  12,  8,  2,
       1,  4,   6,  4,  1};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void sobelV5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
      1,  2, 0,  -2, -1,
      4,  8, 0,  -8, -4,
      6, 12, 0, -12, -6,
      4,  8, 0,  -8, -4,
      1,  2, 0,  -2, -1};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void sobelCross3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1, 0,  1,
       0, 0,  0,
       1, 0, -1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void sobelCross5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
      -1, -2, 0,  2,  1,
      -2, -4, 0,  4,  2,
       0,  0, 0,  0,  0,
       2,  4, 0, -4, -2,
       1,  2, 0, -2, -1};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void sobel3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrixH[9] = {
      -1, -2, -1,
       0,  0,  0,
       1,  2,  1};

   CONST float matrixV[9] = {
      1, 0, -1,
      2, 0, -2,
      1, 0, -1};

   REAL sumH = Convolution(source DIMS, matrixH, 3);
   REAL sumV = Convolution(source DIMS, matrixV, 3);

   REAL Result = Combine(sumH, sumV);

   WRITE(dest, Result);
}

kernel void sobel5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrixH[25] = {
      -1, -4,  -6, -4, -1,
      -2, -8, -12, -8, -2,
       0,  0,   0,  0,  0,
       2,  8,  12,  8,  2,
       1,  4,   6,  4,  1};

   CONST float matrixV[25] = {
      1,  2, 0,  -2, -1,
      4,  8, 0,  -8, -4,
      6, 12, 0, -12, -6,
      4,  8, 0,  -8, -4,
      1,  2, 0,  -2, -1};

   REAL sumH = Convolution(source DIMS, matrixH, 5);
   REAL sumV = Convolution(source DIMS, matrixV, 5);

   WRITE(dest, Combine(sumH, sumV));
}

kernel void prewittH3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1, -1, -1,
       0,  0,  0,
       1,  1,  1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void prewittV3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      1, 0, -1,
      1, 0, -1,
      1, 0, -1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void prewitt3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrixH[9] = {
      -1, -1, -1,
       0,  0,  0,
       1,  1,  1};

   CONST float matrixV[9] = {
      1, 0, -1,
      1, 0, -1,
      1, 0, -1};

   REAL sumH = Convolution(source DIMS, matrixH, 3);
   REAL sumV = Convolution(source DIMS, matrixV, 3);

   WRITE(dest, Combine(sumH, sumV));
}

kernel void scharrH3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -3, -10, -3,
       0,   0,  0,
       3,  10,  3};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void scharrV3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
       -3, 0,  3,
      -10, 0, 10,
       -3, 0,  3};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void scharr3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrixH[9] = {
      -3, -10, -3,
       0,   0,  0,
       3,  10,  3};

   CONST float matrixV[9] = {
       -3, 0,  3,
      -10, 0, 10,
       -3, 0,  3};

   REAL sumH = Convolution(source DIMS, matrixH, 3);
   REAL sumV = Convolution(source DIMS, matrixV, 3);

   WRITE(dest, Combine(sumH, sumV));
}

kernel void hipass3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1, -1, -1,
      -1,  8, -1,
      -1, -1, -1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void hipass5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, 24, -1, -1,
      -1, -1, -1, -1, -1,
      -1, -1, -1, -1, -1};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void laplace3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1, -1, -1,
      -1,  8, -1,
      -1, -1, -1};

   convolution(source, dest DIMS, matrix, 3);
}

kernel void laplace5(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[25] = {
      -1, -3, -4, -3, -1,
      -3,  0,  6,  0, -3,
      -4,  6, 20,  6, -4,
      -3,  0,  6,  0, -3,
      -1, -3, -4, -3, -1};

   convolution(source, dest DIMS, matrix, 5);
}

kernel void sharpen3(INPUT source, OUTPUT dest DIM_ARGS)
{
   CONST float matrix[9] = {
      -1.f/8, -1.f/8, -1.f/8,
      -1.f/8, 16.f/8, -1.f/8,
      -1.f/8, -1.f/8, -1.f/8};

   convolution(source, dest DIMS, matrix, 3);
}

// Smooth convolution (box filter - a convolution matrix filled with 1/Nb)
#undef CONVOLUTION_CASE
#define CONVOLUTION_CASE(size)\
   case size:\
   {\
      for (int y = -size; y <= size; y++)\
         for (int x = -size; x <= size; x++)\
            sum += factor * READ(source, pos + (int2)(x, y));\
   }\
   break;

kernel void smooth(INPUT source, OUTPUT dest DIM_ARGS, int matrix_width)    // Box filter
{
   BEGIN

   const int mask_size = matrix_width / 2;
   REAL sum = 0;
   REAL factor = 1.f / (matrix_width * matrix_width);

   if (OutsideImage(pos DIMS, mask_size))
      return;

   CONVOLUTION_SWITCH

   WRITE(dest, sum);
}


// Median

#define LW 16  // Cached version of median filter uses a 16x16 local size

REAL calculate_median3 (REAL * values)
{
   REAL Tmp;

   // Starting with a subset of size 6, remove the min and max each time
   mnmx6(0,1,2,3,4,5);
   mnmx5(1,2,3,4,6);
   mnmx4(2,3,4,7);
   mnmx3(3,4,8);
   
   return values[4];
}

REAL calculate_median5 (REAL * values)
{
   REAL Tmp;

   mnmx14(0,1,2,3,4,5,6,7,8,9,10,11,12,13);
   mnmx13(1,2,3,4,5,6,7,8,9,10,11,12,14);
   mnmx12(2,3,4,5,6,7,8,9,10,11,12,15);
   mnmx11(3,4,5,6,7,8,9,10,11,12,16);
   mnmx10(4,5,6,7,8,9,10,11,12,17);
   mnmx9( 5,6,7,8,9,10,11,12,18);
   mnmx8( 6,7,8,9,10,11,12,19);
   mnmx7( 7,8,9,10,11,12,20);
   mnmx6( 8,9,10,11,12,21);
   mnmx5( 9,10,11,12,22);
   mnmx4( 10,11,12,23);
   mnmx3( 11,12,24);

   return values[12];
}


#undef  MW
#define MW 3   // Matrix width
__attribute__((reqd_work_group_size(LW, LW, 1)))
kernel void median3_cached (INPUT source, OUTPUT dest DIM_ARGS)
{
   BEGIN

   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);

   // Cache pixels
   local REAL cache[LW * LW];
   cache[lid] = READ(source, pos);
   barrier(CLK_LOCAL_MEM_FENCE);

   // Cache information
   int bitmask = LW - 1;
   int x_cache_begin = gx - (gx & bitmask);
   int x_cache_end = x_cache_begin + LW;
   int y_cache_begin = gy - (gy & bitmask);
   int y_cache_end = y_cache_begin + LW;

   const int matrix_width = MW;
   const int mask_size = matrix_width / 2;

   if (OutsideImage(pos DIMS, mask_size))
      return;

   // Read values in a local array
   REAL values[MW * MW];

   int Index = 0;
   for (int y = -mask_size; y <= mask_size; y++)
   {
      int py = gy + y;
      if (py < y_cache_begin || py >= y_cache_end)
      {
         for (int x = -mask_size; x <= mask_size; x++)
            values[Index++] = READ(source, pos + (int2)(x, y));
      }
      else
      {
         for (int x = -mask_size; x <= mask_size; x++)
         {
            int px = gx + x;
            if (px < x_cache_begin || px >= x_cache_end)
               values[Index++] = READ(source, (int2)(px, py));
            else
            {
               // Read from cache
               int cache_y = py - y_cache_begin;
               int cache_x = px - x_cache_begin;
               values[Index++] = cache[cache_y * LW + cache_x];
            }

         }

      }

   }

   // Calculate median
   REAL Result = calculate_median3 (values);
   
   // Save result
   WRITE(dest, Result);
}

kernel void median3 (INPUT source, OUTPUT dest DIM_ARGS)
{
   BEGIN

   const int matrix_width = MW;
   const int mask_size = matrix_width / 2;

   if (OutsideImage(pos DIMS, mask_size))
      return;

   // Read values in a local array
   REAL values[MW * MW];

   int Index = 0;
   for (int y = -mask_size; y <= mask_size; y++)
      for (int x = -mask_size; x <= mask_size; x++)
         values[Index++] = READ(source, pos + (int2)(x, y));

   // Calculate median
   REAL Result = calculate_median3 (values);
   
   // Save result
   WRITE(dest, Result);
}


#undef MW
#define MW 5   // Matrix width

kernel void median5 (INPUT source, OUTPUT dest DIM_ARGS)
{
   BEGIN

   const int matrix_width = MW;
   const int mask_size = matrix_width / 2;

   if (OutsideImage(pos DIMS, mask_size))
      return;

   // Read values in a local array
   REAL values[MW * MW];

   int Index = 0;
   for (int y = -mask_size; y <= mask_size; y++)
      for (int x = -mask_size; x <= mask_size; x++)
         values[Index++] = READ(source, pos + (int2)(x, y));

   // Calculate median
   REAL Result = calculate_median5 (values);
   
   // Save result
   WRITE(dest, Result);
}
