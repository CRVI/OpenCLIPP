////////////////////////////////////////////////////////////////////////////////
//! @file	: Histogram.cl
//! @date   : Jul 2013
//!
//! @brief  : Histogram calculation on images
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


#define HIST_SIZE 256

// Histogram on a 1 channel image that has a range of 0-255
//    hist must be an array of 256 32b integers, all initialized to 0
kernel void histogram_1C(INPUT source, global uint * hist)
{
   BEGIN

   if (get_local_size(0) * get_local_size(1) < HIST_SIZE)   // Workgroup size is too small - use a workgroup size of at least 256
      return;

   local uint local_hist[HIST_SIZE];

   int local_index = get_local_id(0) + get_local_id(1) * get_local_size(0);

   // Initialize local histogram to 0
   if (local_index < HIST_SIZE)
      local_hist[local_index] = 0;

   // Synch local threads
   barrier(CLK_LOCAL_MEM_FENCE);

   // Read pixel
   uint4 color = READ_IMAGE(source, pos);

   // Increment histogram value
   if (color.x < HIST_SIZE)
      atom_inc(&local_hist[color.x]);

   // Synch local threads
   barrier(CLK_LOCAL_MEM_FENCE);

   // Add all local values to global histogram
   if (local_index < HIST_SIZE)
      atom_add(&hist[local_index], local_hist[local_index]);
}

// Histogram on a 4 channel image that has a range of 0-255
//    hist must be an array of 256*4 32b integers, all initialized to 0
kernel void histogram_4C(INPUT source, global uint * hist)
{
   BEGIN

   if (get_local_size(0) * get_local_size(1) < HIST_SIZE)   // Workgroup size is too small - use a workgroup size of at least 256
      return;

   local uint local_hist1[HIST_SIZE], local_hist2[HIST_SIZE], local_hist3[HIST_SIZE], local_hist4[HIST_SIZE];

   int local_index = get_local_id(0) + get_local_id(1) * get_local_size(0);

   // Initialize local histogram to 0
   if (local_index < HIST_SIZE)
   {
      local_hist1[local_index] = 0;
      local_hist2[local_index] = 0;
      local_hist3[local_index] = 0;
      local_hist4[local_index] = 0;
   }

   // Synch local threads
   barrier(CLK_LOCAL_MEM_FENCE);

   // Read pixel
   uint4 color = READ_IMAGE(source, pos);

   // Increment histogram values
   if (color.x < HIST_SIZE)
      atom_inc(&local_hist1[color.x]);

   if (color.y < HIST_SIZE)
      atom_inc(&local_hist2[color.y]);

   if (color.z < HIST_SIZE)
      atom_inc(&local_hist3[color.z]);

   if (color.w < HIST_SIZE)
      atom_inc(&local_hist4[color.w]);

   // Synch local threads
   barrier(CLK_LOCAL_MEM_FENCE);

   // Add all local values to global histogram
   if (local_index < HIST_SIZE)
   {
      atom_add(&hist[local_index + 0 * HIST_SIZE], local_hist1[local_index]);
      atom_add(&hist[local_index + 1 * HIST_SIZE], local_hist2[local_index]);
      atom_add(&hist[local_index + 2 * HIST_SIZE], local_hist3[local_index]);
      atom_add(&hist[local_index + 3 * HIST_SIZE], local_hist4[local_index]);
   }

}
