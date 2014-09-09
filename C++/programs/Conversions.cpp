////////////////////////////////////////////////////////////////////////////////
//! @file	: Conversions.cpp
//! @date   : Jun 2014
//!
//! @brief  : Image depth conversion
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

#include "Programs/Conversions.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

double GetTypeRange(const Image& Img);
double GetTypeMin(const Image& Img);

// Copy & Convert

void Conversions::Convert(Image& Source, Image& Dest)
{
   CheckSameSize(Source, Dest);

   if (SameType(Source, Dest))
   {
      Copy(Source, Dest);
      return;
   }

   switch (Dest.DataType())
   {
   case SImage::U8:
      Kernel(to_uchar, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::S8:
      Kernel(to_char, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::U16:
      Kernel(to_ushort, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::S16:
      Kernel(to_short, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::U32:
      Kernel(to_uint, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::S32:
      Kernel(to_int, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::F32:
      Kernel(to_float, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::F64:
      Kernel(to_double, Source, Dest, Source.Step(), Dest.Step());
      break;
   case SImage::NbDataTypes:
   default:
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Unsupported data type");
   }
   
}

void Conversions::Scale(Image& Source, Image& Dest)
{
   CheckSameSize(Source, Dest);

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

void Conversions::Scale(Image& Source, Image& Dest, int Offset, float Ratio)
{
   CheckSameSize(Source, Dest);

   switch (Dest.DataType())
   {
   case SImage::U8:
      Kernel(scale_to_uchar, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::S8:
      Kernel(scale_to_char, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::U16:
      Kernel(scale_to_ushort, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::S16:
      Kernel(scale_to_short, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::U32:
      Kernel(scale_to_uint, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::S32:
      Kernel(scale_to_int, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::F32:
      Kernel(scale_to_float, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::F64:
      Kernel(scale_to_double, Source, Dest, Source.Step(), Dest.Step(), Offset, Ratio);
      break;
   case SImage::NbDataTypes:
   default:
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Unsupported data type");
   }

}

void Conversions::Copy(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   if (Source.Step() == Dest.Step())
   {
      Source.SendIfNeeded();

      m_CL->GetQueue().enqueueCopyBuffer(Source, Dest, 0, 0, Source.Step() * Source.Height());

      Dest.SetInDevice();

      return;
   }

   Kernel(copy, Source, Dest, Source.Step(), Dest.Step());
}

void Conversions::ToGray(Image& Source, Image& Dest)
{
   CheckSizeAndType(Source, Dest);
   Check1Channel(Dest);

   Kernel(to_gray, Source, Dest);
}

void Conversions::SelectChannel(Image& Source, Image& Dest, int ChannelNo)
{
   CheckSizeAndType(Source, Dest);
   Check1Channel(Dest);

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
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Unsupported number of channels");
   }
   
}

void Conversions::ToColor(Image& Source, Image& Dest)
{
   CheckSizeAndType(Source, Dest);
   Check1Channel(Source);

   switch (Dest.NbChannels())
   {
   case 1:
      Copy(Source, Dest);
      break;
   case 2:
      Kernel(to_2channels, Source, Dest, Source.Step(), Dest.Step());
      break;
   case 3:
      Kernel(to_3channels, Source, Dest, Source.Step(), Dest.Step());
      break;
   case 4:
      Kernel(to_4channels, Source, Dest, Source.Step(), Dest.Step());
      break;
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Wrong ChannelNo in Conversions::SelectChannel() - allowed values : 1 to 4");
   }

   // select_channel1 does what we want
   Kernel(select_channel1, Source, Dest);
}

void Conversions::Copy3Cto4C(Image& Source, Image& Dest)
{
   CheckSizeAndType(Source, Dest);

   if (Source.NbChannels() != 3)
      throw cl::Error(CL_INVALID_VALUE, "Source image must be 3 channels in Copy3Cto4C");

   if (Dest.NbChannels() != 4)
      throw cl::Error(CL_INVALID_VALUE, "Dest image must be 4 channels in Copy3Cto4C");

   Kernel(copy3Cto4C, Source, Dest, Source.Step(), Dest.Step());
}

void Conversions::Copy4Cto3C(Image& Source, Image& Dest)
{
   CheckSizeAndType(Source, Dest);

   if (Source.NbChannels() != 4)
      throw cl::Error(CL_INVALID_VALUE, "Source image must be 4 channels in Copy4Cto3C");

   if (Dest.NbChannels() != 3)
      throw cl::Error(CL_INVALID_VALUE, "Dest image must be 3 channels in Copy4Cto3C");

   Kernel(copy4Cto3C, Source, Dest, Source.Step(), Dest.Step());
}


// Helpers

double GetTypeRange(const Image& Img)
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

double GetTypeMin(const Image& Img)
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
