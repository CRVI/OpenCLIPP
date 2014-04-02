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

#include "Buffers.h"

#define WIDTH1    8    // Number of pixels processed by 1 work item
#define WG_WIDTH  8
#define WG_HEIGHT 8
#define WG_SIZE (WG_WIDTH * WG_HEIGHT)

//----------------------------------------------------------------------------------------------------------------------

// 32 bits Local sqr (in an area of 256*16)
__attribute__((reqd_work_group_size(WG_WIDTH, WG_HEIGHT, 1)))
kernel void sqr_F32(INPUT_SPACE const SCALAR * source, global float * dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(float);
   const int gx = get_global_id(0) * WIDTH1;
   const int gy = get_global_id(1);
   const int local_x = get_local_id(0) * WIDTH1;
   const int local_y = get_local_id(1);
   const int buf_index = local_x + local_y * WG_WIDTH * WIDTH1;
   
   local float TmpSumX[WG_SIZE * WIDTH1];
   local float SumX[WG_SIZE * WIDTH1];

   // 16 pixel wide summation on X
   float Sum = 0;
   for (int i = 0; i < WIDTH1; i++)
   {
	  float v = (float)source[gy * src_step + gx + i];
      Sum += v * v;
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
      float Val = Sum + TmpSumX[buf_index + i];
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
		 dest[(get_group_id(1) * WG_HEIGHT + i) * dst_step + gx2] = Sum;
   }

}

// 32 bits Local scan (in an area of 256*16)
__attribute__((reqd_work_group_size(WG_WIDTH, WG_HEIGHT, 1)))
kernel void scan1_F32(INPUT_SPACE const SCALAR * source, global float * dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(float);
   const int gx = get_global_id(0) * WIDTH1;
   const int gy = get_global_id(1);
   const int local_x = get_local_id(0) * WIDTH1;
   const int local_y = get_local_id(1);
   const int buf_index = local_x + local_y * WG_WIDTH * WIDTH1;
   
   local float TmpSumX[WG_SIZE * WIDTH1];
   local float SumX[WG_SIZE * WIDTH1];

   // 16 pixel wide summation on X
   float Sum = 0;
   for (int i = 0; i < WIDTH1; i++)
   {
      Sum += (float)source[gy * src_step + gx + i];
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
      float Val = Sum + TmpSumX[buf_index + i];
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
		 dest[(get_group_id(1) * WG_HEIGHT + i) * dst_step + gx2] = Sum;
   }

}

// 32 bits Vertical scan - reads the last pixel of each row of each group
kernel void scan2_F32(global float * dest, global float * vert, int dest_step, int vert_step)
{
   dest_step /= sizeof(float);
   vert_step /= sizeof(float);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   float Sum = 0;
   for (int i = 0; i <= gx; i++)
   {
      float v = dest[gy * dest_step + (i + 1) * WG_WIDTH * WIDTH1 - 1];
      Sum += v;  // Sum value of pixels to the left
   }

   vert[gy * vert_step + gx] = Sum;
}

// 32 bits Vertical combine - adds values from the left of the group
kernel void scan3_F32(global float * dest_in, global float * vert, global float * dest, int dest_in_step, int vert_step, int dest_step)
{
   dest_in_step /= sizeof(float);
   dest_step /= sizeof(float);
   vert_step /= sizeof(float);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int tx = (get_global_id(0) / (WIDTH1 * WG_WIDTH)) - 1;

   if (tx < 0)
      return;  // Left-most groups have nothing to add

   // Read sum of other groups
   float Outside = vert[gy * vert_step + tx];

   // Read partial sum for this pixel
   float Partial = dest_in[gy * dest_in_step + gx];

   // Write final value
   dest[gy * dest_step + gx] = Partial + Outside;
}

// Horizontal scan - reads the last pixel of each row of each group
kernel void scan4_F32(global float * dest, global float * horiz, int dest_step, int horiz_step)
{
   dest_step /= sizeof(float);
   horiz_step /= sizeof(float);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   float Sum = 0;
   for (int i = 0; i <= gy; i++)
   {
	  float v = dest[((i + 1) * WG_HEIGHT - 1) * dest_step + gx];
	  Sum += v;// Sum value of pixels to the top
   }
   horiz[gy * horiz_step + gx] = Sum;
}

// 32 bits Horizontal combine - adds values from the top of the group
kernel void scan5_F32(global float * dest_in, global float * horiz, global float * dest, int dest_in_step, int horiz_step, int dest_step)
{
   dest_in_step /= sizeof(float);
   dest_step /= sizeof(float);
   horiz_step /= sizeof(float);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int ty = (get_global_id(1) / WG_HEIGHT) - 1;

   if (ty < 0)
      return;  // Top-most groups have nothing to add

   // Read sum of other groups
   float Outside = horiz[ty * horiz_step + gx];

   // Read partial sum for this pixel
   float Partial = dest_in[gy * dest_in_step + gx];

   // Write final value
   dest[gy * dest_step + gx] = Partial + Outside;
}

//----------------------------------------------------------------------------------------------------------------------

// Local sqr (in an area of 256*16)
__attribute__((reqd_work_group_size(WG_WIDTH, WG_HEIGHT, 1)))
kernel void sqr_F64(INPUT_SPACE const SCALAR * source, global double * dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(double);
   const int gx = get_global_id(0) * WIDTH1;
   const int gy = get_global_id(1);
   const int local_x = get_local_id(0) * WIDTH1;
   const int local_y = get_local_id(1);
   const int buf_index = local_x + local_y * WG_WIDTH * WIDTH1;
   
   local double TmpSumX[WG_SIZE * WIDTH1];
   local double SumX[WG_SIZE * WIDTH1];

   // 16 pixel wide summation on X
   double Sum = 0;
   for (int i = 0; i < WIDTH1; i++)
   {
	  double v = (double)source[gy * src_step + gx + i];
      Sum += v * v;
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
      double Val = Sum + TmpSumX[buf_index + i];
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
		 dest[(get_group_id(1) * WG_HEIGHT + i) * dst_step + gx2] = Sum;
   }

}

// Local scan (in an area of 256*16)
__attribute__((reqd_work_group_size(WG_WIDTH, WG_HEIGHT, 1)))
kernel void scan1_F64(INPUT_SPACE const SCALAR * source, global double * dest, int src_step, int dst_step, int width, int height)
{
   src_step /= sizeof(SCALAR);
   dst_step /= sizeof(double);
   const int gx = get_global_id(0) * WIDTH1;
   const int gy = get_global_id(1);
   const int local_x = get_local_id(0) * WIDTH1;
   const int local_y = get_local_id(1);
   const int buf_index = local_x + local_y * WG_WIDTH * WIDTH1;
   
   local double TmpSumX[WG_SIZE * WIDTH1];
   local double SumX[WG_SIZE * WIDTH1];

   // 16 pixel wide summation on X
   double Sum = 0;
   for (int i = 0; i < WIDTH1; i++)
   {
      Sum += source[gy * src_step + gx + i];
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
      double Val = Sum + TmpSumX[buf_index + i];
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
		 dest[(get_group_id(1) * WG_HEIGHT + i) * dst_step + gx2] = Sum;
   }

}

// Vertical scan - reads the last pixel of each row of each group
kernel void scan2_F64(global double * dest, global double * vert, int dest_step, int vert_step)
{
   dest_step /= sizeof(double);
   vert_step /= sizeof(double);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   double Sum = 0;
   for (int i = 0; i <= gx; i++)
   {
      double v = dest[gy * dest_step + (i + 1) * WG_WIDTH * WIDTH1 - 1];
      Sum += v;  // Sum value of pixels to the left
   }

   vert[gy * vert_step + gx] = Sum;
}

// Vertical combine - adds values from the left of the group
kernel void scan3_F64(global double * dest_in, global double * vert, global double * dest, int dest_in_step, int vert_step, int dest_step)
{
   dest_in_step /= sizeof(double);
   dest_step /= sizeof(double);
   vert_step /= sizeof(double);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int tx = (get_global_id(0) / (WIDTH1 * WG_WIDTH)) - 1;

   if (tx < 0)
      return;  // Left-most groups have nothing to add

   // Read sum of other groups
   double Outside = vert[gy * vert_step + tx];

   // Read partial sum for this pixel
   double Partial = dest_in[gy * dest_in_step + gx];

   // Write final value
   dest[gy * dest_step + gx] = Partial + Outside;
}

// Horizontal scan - reads the last pixel of each row of each group
kernel void scan4_F64(global double * dest, global double * horiz, int dest_step, int horiz_step)
{
   dest_step /= sizeof(double);
   horiz_step /= sizeof(double);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);

   double Sum = 0;
   for (int i = 0; i <= gy; i++)
   {
	  double v = dest[((i + 1) * WG_HEIGHT - 1) * dest_step + gx];
	  Sum += v;// Sum value of pixels to the top
   }
   horiz[gy * horiz_step + gx] = Sum;
}

// Horizontal combine - adds values from the top of the group
kernel void scan5_F64(global double * dest_in, global double * horiz, global double * dest, int dest_in_step, int horiz_step, int dest_step)
{
   dest_in_step /= sizeof(double);
   dest_step /= sizeof(double);
   horiz_step /= sizeof(double);

   const int gx = get_global_id(0);
   const int gy = get_global_id(1);
   const int ty = (get_global_id(1) / WG_HEIGHT) - 1;

   if (ty < 0)
      return;  // Top-most groups have nothing to add

   // Read sum of other groups
   double Outside = horiz[ty * horiz_step + gx];

   // Read partial sum for this pixel
   double Partial = dest_in[gy * dest_in_step + gx];

   // Write final value
   dest[gy * dest_step + gx] = Partial + Outside;
}