////////////////////////////////////////////////////////////////////////////////
//! @file	: Histogram.cpp
//! @date   : Jun 2014
//!
//! @brief  : Histogram calculation on images
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

#include "Programs/Histogram.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{


// Histogram must be an array of at least 256 elements
void Histogram::Histogram1C(Image& Source, uint * Histogram)
{
   const static int Length = 256;

   for (int i = 0; i < Length; i++)
      Histogram[i] = 0;

   Buffer Buffer(*m_CL, Histogram, Length);
   Buffer.Send();

   Kernel(histogram_1C, In(Source), Out(), Buffer);

   Buffer.Read(true);
}

// Histogram must be an array of at least 1024 elements
void Histogram::Histogram4C(Image& Source, uint * Histogram)
{
   const static int Length = 256 * 4;

   for (int i = 0; i < Length; i++)
      Histogram[i] = 0;

   Buffer Buffer(*m_CL, Histogram, Length);
   Buffer.Send();

   Kernel(histogram_4C, In(Source), Out(), Buffer);

   Buffer.Read(true);
}

uint Histogram::OtsuThreshold(uint Histogram[256], uint NbPixels)
{
   double sum = 0;
   for (int i = 0; i < 256; i++)
      sum += i * Histogram[i];

   int wB = 0;
   int wF = 0;
   uint threshold = 0;
   double sumB = 0;
   double maxVariance = 0;
   for (uint i = 0; i < 256; i++)
   {
      wB += Histogram[i];  // Weight background
      
      if (wB == 0)
         continue;

      wF = int(NbPixels) - wB;  // Weight foreground
      
      if (wF == 0)
         break;

      sumB += i * Histogram[i];
      
      double mB = sumB / wB;  // Mean background
      double mF = (sum - sumB) / wF;   // Mean foreground
      
      double variance = wB * wF * (mB - mF) * (mB - mF);
      
      if (variance > maxVariance)
      {
         maxVariance = variance;
         threshold = i;
      }
      
   }

   return threshold;
}

uint Histogram::OtsuThreshold(Image& Source)
{
   uint Histogram[256];
   Histogram1C(Source, Histogram);

   return OtsuThreshold(Histogram, Source.Width() * Source.Height());
}

}
