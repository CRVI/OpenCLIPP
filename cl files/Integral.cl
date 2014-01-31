////////////////////////////////////////////////////////////////////////////////
//! @file	: Integral.cl
//! @date   : Jul 2013
//!
//! @brief  : Calculates the integral sum scan of an image
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


// CLK_ADDRESS_CLAMP to receive 0 when outside of image
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;


#ifdef I

   // For signed integer images
   #define READ_IMAGE(img, pos) read_imagei(img, sampler, pos).x
   //#define WRITE_IMAGE(img, pos, px) write_imagei(img, pos, (int4)(px, 0, 0, 0))
   #define TYPE int

#else // I

   #ifdef UI

      // For unsigned integer images
      #define READ_IMAGE(img, pos) read_imageui(img, sampler, pos).x
      //#define WRITE_IMAGE(img, pos, px) write_imageui(img, pos, (uint4)(px, 0, 0, 0))
      #define TYPE uint

   #else // UI

      // For float
      #define READ_IMAGE(img, pos) read_imagef(img, sampler, pos).x
      
      #define TYPE float

   #endif // UI

#endif // I

#define WRITE_IMAGE(img, pos, px) write_imagef(img, pos, (float4)(px, 0, 0, 0))

#define WIDTH1    16    // Number of pixels processed by 1 work item
#define WG_WIDTH  16
#define WG_HEIGHT 16
#define WG_SIZE (WG_WIDTH * WG_HEIGHT)


// Local scan (in an area of 256*16)
__attribute__((reqd_work_group_size(WG_WIDTH, WG_HEIGHT, 1)))
kernel void scan1(read_only image2d_t source, write_only image2d_t dest, uint width, uint height)
{
   const int gx = get_global_id(0) * WIDTH1;
   const int gy = get_global_id(1);
   const int local_x = get_local_id(0) * WIDTH1;
   const int local_y = get_local_id(1);
   const int buf_index = local_x + local_y * WG_WIDTH * WIDTH1;
   
   local TYPE TmpSumX[WG_SIZE * WIDTH1];
   local TYPE SumX[WG_SIZE * WIDTH1];

   // 16 pixel wide summation on X
   TYPE Sum = 0;
   for (int i = 0; i < WIDTH1; i++)
   {
      Sum += READ_IMAGE(source, (int2)(gx + i, gy));
      TmpSumX[buf_index + i] = Sum;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   // Get the sum of preceding groups of 16 pixels
   Sum = 0;
   for (int x = WIDTH1 - 1; x < local_x; x += WIDTH1)
      Sum += TmpSumX[local_y * WG_WIDTH * WIDTH1 + x];

   // Add to the summation
   for (int i = 0; i < WIDTH1; i++)
   {
      TYPE Val = Sum + TmpSumX[buf_index + i];
      SumX[buf_index + i] = Val;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   // We now have a full summation on X of 256 pixels

   // Now sum Y values
   // NOTE : This part assumes that WG_HEIGHT == WG_WIDTH == WIDTH1

   const int gx2 = gx + local_y;

   if (gx2 >= width)
      return;

   Sum = 0;
   for (int i = 0; i < WG_HEIGHT; i++)
   {
      int2 Pos = {gx2, get_group_id(1) * WG_HEIGHT + i};

      Sum += SumX[i * WG_WIDTH * WIDTH1 + local_x + local_y];

      if (Pos.y < height)
         WRITE_IMAGE(dest, Pos, Sum);
   }

}

// Vertical scan - reads the last pixel of each row of each group
kernel void scan2(read_only image2d_t dest, write_only image2d_t vert)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int2 pos = {gx, gy};

   TYPE Sum = 0;
   for (int i = 0; i <= gx; i++)
      Sum += READ_IMAGE(dest, (int2)((i + 1) * WG_WIDTH * WIDTH1 - 1, gy));  // Sum value of pixels to the left

   WRITE_IMAGE(vert, pos, Sum);
}

// Vertical combine - adds values from the left of the group
kernel void scan3(read_only image2d_t dest_in, read_only image2d_t vert, write_only image2d_t dest)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int tx = (get_global_id(0) / (WIDTH1 * WG_WIDTH)) - 1;
   const int2 tpos = {tx, gy};
   const int2 Pos = {gx, gy};

   if (tx < 0)
      return;  // Left-most groups have nothing to add

   // Read sum of other groups
   TYPE Outside = READ_IMAGE(vert, tpos);

   // Read partial sum for this pixel
   TYPE Partial = READ_IMAGE(dest_in, Pos);

   // Write final value
   WRITE_IMAGE(dest, Pos, Partial + Outside);
}

// Horizontal scan - reads the last pixel of each row of each group
kernel void scan4(read_only image2d_t dest, write_only image2d_t horiz)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int2 pos = {gx, gy};

   TYPE Sum = 0;
   for (int i = 0; i <= gy; i++)
      Sum += READ_IMAGE(dest, (int2)(gx, (i + 1) * WG_HEIGHT - 1));  // Sum value of pixels to the top

   WRITE_IMAGE(horiz, pos, Sum);
}

// Horizontal combine - adds values from the top of the group
kernel void scan5(read_only image2d_t dest_in, read_only image2d_t horiz, write_only image2d_t dest)
{
   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int ty = (get_global_id(1) / WG_HEIGHT) - 1;
   const int2 tpos = {gx, ty};
   const int2 Pos = {gx, gy};

   if (ty < 0)
      return;  // Top-most groups have nothing to add

   // Read sum of other groups
   TYPE Outside = READ_IMAGE(horiz, tpos);

   // Read partial sum for this pixel
   TYPE Partial = READ_IMAGE(dest_in, Pos);

   // Write final value
   WRITE_IMAGE(dest, Pos, Partial + Outside);
}
