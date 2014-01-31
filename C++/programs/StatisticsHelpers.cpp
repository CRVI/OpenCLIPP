////////////////////////////////////////////////////////////////////////////////
//! @file	: StatisticsHelpers.cpp
//! @date   : Sep 2013
//!
//! @brief  : Tools to help implementation of reduction-type programs
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

// Helpers for Statistics reduction

// Local group & global group logic :
//  The reduction algorithm expects to have each local group filled with workers (full grids of 16x16)
//  Each worker reads 16 pixels along X (so one group reads a rectangle of 16x16x16 (4096) pixels)
//  Local size is 16x16
//  So for images that have a Width not a multiple of 16*32 or a height not a multiple of 16 :
//    Additional workers are started that only set in_image[lid] to false
//  For images with flush width and height, the faster _flush version is used
//  Each group will work on at least 1 pixel (worst case), so m_Result will be filled with valid data


#include "StatisticsHelpers.h"

namespace OpenCLIPP
{

std::string SelectName(const char * name, const ImageBase& Image)
{
   std::string Name = name;

   if (IsFlushImage(Image))
      Name += "_flush";       // Use faster version

   return Name;
}

double ReduceSum(std::vector<float>& buffer)
{
   // First half of buffer contains the sums
   // Second half of buffer contains the number of pixels per sum (which we don't need)
   size_t size = buffer.size() / 2;

   double Sum = buffer[0];
   for (size_t i = 1; i < size; i++)
      Sum += buffer[i];

   return Sum;
}

double ReduceMean(std::vector<float>& buffer)
{
   // First half of buffer contains the mean of all pixels
   // Second half of buffer contains the number of pixels used to generate the mean
   size_t size = buffer.size() / 2;

   double MeanSum = buffer[0];      // MeanSum will contain the sum of the means
   double NormalNb = buffer[size];  // We use the number of pixels from the first workgroup as reference
   double SumPixels = buffer[size]; // Will be equal to the number of pixels in the image
   for (size_t i = 1; i < size; i++)
   {
      double NbPixels = buffer[size + i];
      double Ratio = NbPixels / NormalNb; // If this workgroup had less pixels, Ratio will be smaller 

      MeanSum += buffer[i] * Ratio;

      SumPixels += NbPixels;
   }

   double Divisor = SumPixels / NormalNb; // Normalize the divisor

   return MeanSum / Divisor;  // Divide the sum to get the final mean
}

}
