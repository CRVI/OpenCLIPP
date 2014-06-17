////////////////////////////////////////////////////////////////////////////////
//! @file	: Filters.cpp
//! @date   : Jan 2014
//!
//! @brief  : Convolution-type filters
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

#include "Programs/Filters.h"
#include <vector>

#include "kernel_helpers.h"

#include <math.h>

namespace OpenCLIPP
{

static void GenerateBlurMask(std::vector<float>& Mask, float Sigma, int MaskSize)
{
   float sum = 0;

   for(int x = -MaskSize; x <= MaskSize; x++)
      for(int y = -MaskSize; y <= MaskSize; y++)
      {
         float temp = exp(-(float(x * x + y * y) / (2 * Sigma * Sigma)));
         sum += temp;
         Mask[x + MaskSize + (y + MaskSize) * (MaskSize * 2 + 1)] = temp;
      }

    // Normalize the mask
    for(float& v : Mask)
        v /= sum;
}

void Filters::GaussianBlur(Image& Source, Image& Dest, float Sigma)
{
   CheckCompatibility(Source, Dest);

   // Prepare mask
   int MaskSize = int(ceil(3 * Sigma));

   if (Sigma <= 0 || MaskSize > 31)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid sigma used with GaussianBlur - allowed : 0.01-10");

   uint NbElements = (MaskSize * 2 + 1 ) * (MaskSize * 2 + 1 );

   std::vector<float> Mask(NbElements);

   GenerateBlurMask(Mask, Sigma, MaskSize);
   // NOTE : Maybe we should generate the mask in the device to prevent having to send that buffer


   // Send mask to device
   ReadBuffer MaskBuffer(*m_CL, Mask.data(), NbElements);

   // Execute kernel
   Kernel(gaussian_blur, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width(), Source.Height(), MaskBuffer, MaskSize);
}

void Filters::Gauss(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(gaussian3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(gaussian5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Gauss - allowed : 3, 5");
}

void Filters::Sharpen(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width != 3)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Sharpen - allowed : 3");

   Kernel(sharpen3, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Filters::Smooth(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width < 3 || (Width & 1) == 0)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Smooth");

   Kernel(smooth, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width(), Source.Height(), Width);
}

/*static bool RangeFit(const ImageBase& Img, int RangeX, int RangeY)
{
   if (Img.Width() % RangeX != 0)
      return false;

   if (Img.Height() % RangeY != 0)
      return false;

   return true;
}*/

void Filters::Median(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   Source.SendIfNeeded();

   if (Width != 3 && Width != 5)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Median - allowed : 3 or 5");

   if (Width == 3)
   {
      /*if (RangeFit(Source, 16, 16))  // The cached version is slower on my GTX 680
      {
         Kernel_(*m_CL, SelectProgram(Source), median3_cached, Source.FullRange(), cl::NDRange(16, 16, 1), Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
         return;
      }*/

      Kernel(median3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   Kernel(median5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Filters::SobelVert(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(sobelV3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(sobelV5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in SobelVert - allowed : 3, 5");
}

void Filters::SobelHoriz(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(sobelH3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(sobelH5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }
   
   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in SobelHoriz - allowed : 3, 5");
}

void Filters::SobelCross(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(sobelCross3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(sobelCross5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }
   
   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in SobelCross - allowed : 3, 5");
}

void Filters::Sobel(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(sobel3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(sobel5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }
   
   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Sobel - allowed : 3, 5");
}

void Filters::PrewittVert(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(prewittV3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in PrewittVert - allowed : 3");
}

void Filters::PrewittHoriz(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(prewittH3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in PrewittHoriz - allowed : 3");
}

void Filters::Prewitt(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(prewitt3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Prewitt - allowed : 3");
}

void Filters::ScharrVert(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(scharrV3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in ScharrVert - allowed : 3");
}

void Filters::ScharrHoriz(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(scharrH3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in ScharrHoriz - allowed : 3");
}

void Filters::Scharr(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(scharr3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Scharr - allowed : 3");
}

void Filters::Hipass(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(hipass3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(hipass5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Hipass - allowed : 3, 5");
}

void Filters::Laplace(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if (Width == 3)
   {
      Kernel(laplace3, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   if (Width == 5)
   {
      Kernel(laplace5, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
      return;
   }

   throw cl::Error(CL_INVALID_ARG_VALUE, "Invalid width used in Laplace - allowed : 3, 5");
}

}
