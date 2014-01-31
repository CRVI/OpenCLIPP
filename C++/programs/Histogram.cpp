////////////////////////////////////////////////////////////////////////////////
//! @file	: Histogram.cpp
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

#include "Programs/Histogram.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{


// Histogram must be an array of at least 256 elements
void Histogram::Histogram1C(IImage& Source, uint * Histogram)
{
   const static int Length = 256;

   for (int i = 0; i < Length; i++)
      Histogram[i] = 0;

   Buffer Buffer(*m_CL, Histogram, Length);
   Buffer.Send();

   Kernel(histogram_1C, Source, Buffer);

   Buffer.Read(true);
}

// Histogram must be an array of at least 1024 elements
void Histogram::Histogram4C(IImage& Source, uint * Histogram)
{
   const static int Length = 256 * 4;

   for (int i = 0; i < Length; i++)
      Histogram[i] = 0;

   Buffer Buffer(*m_CL, Histogram, Length);
   Buffer.Send();

   Kernel(histogram_4C, Source, Buffer);

   Buffer.Read(true);
}

uint Histogram::OtsuTreshold(uint Histogram[256], uint NbPixels)
{
   double sum = 0;
   for (int i = 0; i < 256; i++)
      sum += i * Histogram[i];

   int wB = 0;
   int wF = 0;
   uint treshold = 0;
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
         treshold = i;
      }
      
   }

   return treshold;
}

uint Histogram::OtsuTreshold(IImage& Source)
{
   uint Histogram[256];
   Histogram1C(Source, Histogram);

   return OtsuTreshold(Histogram, Source.Width() * Source.Height());
}

}
