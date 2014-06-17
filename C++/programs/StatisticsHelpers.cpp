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
//  The reduction algorithm expects to have each local group filled with workers (full grids of 16x16 - LWxLW)
//  Each worker reads 16 pixels along X (WIDTH1) (so one group reads a rectangle of 256x16 (4096) pixels)
//  Local size is 16x16
//  So for images that have a Width not a multiple of 256 (LW*WIDTH1) or a height not a multiple of 16 :
//    Additional workers are started that only set nb_pixels[lid] to 0
//  For images with flush width and height, the faster _flush version is used
//  Each group will work on at least 1 pixel (worst case), so m_Result will be filled with valid data


#include "StatisticsHelpers.h"

namespace OpenCLIPP
{

std::string SelectName(const char * name, const ImageBase& Img)
{
   std::string Name = name;

   if (IsFlushImage(Img))
      Name += "_flush";       // Use faster version

   return Name;
}

double ReduceSum(std::vector<float>& buffer)
{
   double Val = 0;
   ReduceSum(buffer, 1, &Val);
   return Val;
}

double ReduceMean(std::vector<float>& buffer)
{
   double Val = 0;
   ReduceMean(buffer, 1, &Val);
   return Val;
}

void ReduceSum(std::vector<float>& buffer, int NbChannels, double outVal[4])
{
   size_t size = buffer.size() / 5;

   for (int i = 0; i < NbChannels; i++)
      outVal[i] = buffer[i];

   for (size_t i = 1; i < size; i++)
      for (int j = 0; j < NbChannels; j++)
         outVal[j] += buffer[i * 4 + j];
}

void ReduceMean(std::vector<float>& buffer, int NbChannels, double outVal[4])
{
   size_t size = buffer.size() / 5;
   size_t NbIndex = size * 4;       // Index where we can find the number of pixels

   for (int j = 0; j < NbChannels; j++)
      outVal[j] = buffer[j];         // MeanSum will contain the sum of the means

   double NormalNb = buffer[NbIndex];  // We use the number of pixels from the first workgroup as reference
   double SumPixels = buffer[NbIndex]; // Will be equal to the number of pixels in the image
   for (size_t i = 1; i < size; i++)
   {
      double NbPixels = buffer[NbIndex + i];
      double Ratio = NbPixels / NormalNb; // If this workgroup had less pixels, Ratio will be smaller 

      for (int j = 0; j < NbChannels; j++)
         outVal[j] += buffer[i * 4 + j] * Ratio;

      SumPixels += NbPixels;
   }

   double Divisor = SumPixels / NormalNb; // Normalize the divisor

   for (int j = 0; j < NbChannels; j++)
      outVal[j] /= Divisor;  // Divide the sum to get the final mean
}

double ReduceMin(std::vector<float>& buffer, std::vector<int>& coords, int& outX, int& outY)
{
   size_t size = buffer.size() / 5;
   int x = coords[0];
   int y = coords[1];

   float v = buffer[0];
   for (size_t i = 0; i < size; i++)
      if (buffer[i] < v)
      {
         v = buffer[i];
         x = coords[i * 2];
         y = coords[i * 2 + 1];
      }

   outX = x;
   outY = y;

   return v;
}

double ReduceMax(std::vector<float>& buffer, std::vector<int>& coords, int& outX, int& outY)
{
   size_t size = buffer.size() / 5;
   int x = coords[0];
   int y = coords[1];

   float v = buffer[0];
   for (size_t i = 0; i < size; i++)
      if (buffer[i] > v)
      {
         v = buffer[i];
         x = coords[i * 2];
         y = coords[i * 2 + 1];
      }

   outX = x;
   outY = y;

   return v;
}

}
