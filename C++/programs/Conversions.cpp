////////////////////////////////////////////////////////////////////////////////
//! @file	: Conversions.cpp
//! @date   : Jul 2013
//!
//! @brief  : Image depth conversion
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

#include "Programs/Conversions.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

double GetTypeRange(const IImage& Img);
double GetTypeMin(const IImage& Img);

// Copy & Convert

void Conversions::Convert(IImage& Source, IImage& Dest)
{
   if (SameType(Source, Dest))
   {
      Copy(Source, Dest);
      return;
   }
   
   if (Dest.IsFloat())
   {
      Kernel(to_float, Source, Dest);
      return;
   }

   if (Dest.IsUnsigned())
   {
      Kernel(to_uint, Source, Dest);
      return;
   }
   
   Kernel(to_int, Source, Dest)
}

void Conversions::Scale(IImage& Source, IImage& Dest)
{
   double SrcRange = GetTypeRange(Source);
   double DstRange = GetTypeRange(Dest);

   double SrcMin = GetTypeMin(Source);
   double DstMin = GetTypeMin(Dest);

   if (SrcRange == DstRange && SrcMin == DstMin)
   {
      Convert(Source, Dest);
      return;
   }

   float Ratio = static_cast<float>(DstRange / SrcRange);
   int Offset = static_cast<int>(-SrcMin * Ratio + DstMin);

   Scale(Source, Dest, Offset, Ratio);
}

void Conversions::Scale(IImage& Source, IImage& Dest, int Offset, float Ratio)
{
   if (Dest.IsFloat())
   {
      Kernel(scale_to_float, Source, Dest, float(Offset), Ratio);
      return;
   }

   if (Dest.IsUnsigned())
   {
      Kernel(scale_to_uint, Source, Dest, Offset, Ratio);
      return;
   }

   Kernel(scale_to_int, Source, Dest, Offset, Ratio)
}

void Conversions::Copy(IImage& Source, IImage& Dest)
{
   CheckSimilarity(Source, Dest);

   Source.SendIfNeeded();

   cl::size_t<3> Origin;
   cl::size_t<3> Region;
   Region[0] = Source.Width();
   Region[1] = Source.Height();
   Region[2] = 1;

   m_CL->GetQueue().enqueueCopyImage(Source, Dest, Origin, Origin, Region);

   Dest.SetInDevice();
}

void Conversions::Copy(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   // TODO : Support buffers with different steps

   if (Source.Step() != Dest.Step())
      throw cl::Error(CL_INVALID_VALUE, "When copying image buffers, they must both have the same step");

   Source.SendIfNeeded();

   m_CL->GetQueue().enqueueCopyBuffer(Source, Dest, 0, 0, Source.Step() * Source.Height());

   Dest.SetInDevice();
}

void Conversions::Copy(IImage& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   // TODO : Support unaligned image buffers

   if (Dest.ElementStep() != Dest.Width())
      throw cl::Error(CL_INVALID_VALUE, "When copying an image to an image buffer, the image buffer must not have any padding at the end of the lines");

   Source.SendIfNeeded();

   cl::size_t<3> Origin;
   cl::size_t<3> Region;
   Region[0] = Source.Width();
   Region[1] = Source.Height();
   Region[2] = 1;

   m_CL->GetQueue().enqueueCopyImageToBuffer(Source, Dest, Origin, Region, 0);

   Dest.SetInDevice();
}

void Conversions::Copy(ImageBuffer& Source, IImage& Dest)
{
   CheckSimilarity(Source, Dest);

   // TODO : Support unaligned image buffers

   if (Source.ElementStep() != Source.Width())
      throw cl::Error(CL_INVALID_VALUE, "When copying an image buffer to an image, the image buffer must not have any padding at the end of the lines");

   Source.SendIfNeeded();

   cl::size_t<3> Origin;
   cl::size_t<3> Region;
   Region[0] = Source.Width();
   Region[1] = Source.Height();
   Region[2] = 1;

   m_CL->GetQueue().enqueueCopyBufferToImage(Source, Dest, 0, Origin, Region);

   Dest.SetInDevice();
}

void Conversions::ToGray(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(to_gray, Source, Dest);
}

void Conversions::SelectChannel(IImage& Source, IImage& Dest, int ChannelNo)
{
   CheckCompatibility(Source, Dest);

   switch (ChannelNo)
   {
   case 1:
      Kernel(select_channel1, Source, Dest);
      break;
   case 2:
      Kernel(select_channel2, Source, Dest);
      break;
   case 3:
      Kernel(select_channel3, Source, Dest);
      break;
   case 4:
      Kernel(select_channel4, Source, Dest);
      break;
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Wrong ChannelNo in Conversions::SelectChannel() - allowed values : 1 to 4");
   }
   
}

void Conversions::ToColor(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   // select_channel1 does what we want
   Kernel(select_channel1, Source, Dest);
}


// Helpers

double GetTypeRange(const IImage& Img)
{
   switch (Img.Depth())
   {
   case 8:
      return 0xFF;
   case 16:
      return 0xFFFF;
   case 32:
      if (Img.IsFloat())
         return 0xFF;

      return 0xFFFFFFFF;
   default:
      assert(false); // Wrong type used
      return 1;
   }

}

double GetTypeMin(const IImage& Img)
{
   if (Img.IsUnsigned())
      return 0;

   if (Img.IsFloat())
      return 0;

   switch (Img.Depth())
   {
   case 8:
      return -128.;
   case 16:
      return -32768.;
   case 32:
      return -2147483648.;
   default:
      assert(false); // Wrong type used
      return 0;
   }

}

}
