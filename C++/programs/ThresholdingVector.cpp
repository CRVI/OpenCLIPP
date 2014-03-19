////////////////////////////////////////////////////////////////////////////////
//! @file	: ThresholdingVector.cpp
//! @date   : Jul 2013
//!
//! @brief  : Thresholding operations on image buffer
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

#include "Programs/ThresholdingVector.h"


#define KERNEL_RANGE(src_img) GetRange(src_img)

#include "kernel_helpers.h"

namespace OpenCLIPP
{

std::string SelectName(const char * name, ThresholdingVector::ECompareOperation Op);


void ThresholdingVector::ThresholdGTLT(ImageBuffer& Source, ImageBuffer& Dest, float threshLT, float valueLower, float threshGT, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(thresholdGTLT, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), threshLT, valueLower, threshGT, valueHigher);
}


#undef SELECT_NAME
#define SELECT_NAME(name, src_img) SelectName( #name , Op)

void ThresholdingVector::Threshold(ImageBuffer& Source, ImageBuffer& Dest, float Thresh, float value, ECompareOperation Op)
{
   CheckCompatibility(Source, Dest);

   Kernel(threshold, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), Thresh, value);
}

void ThresholdingVector::Threshold(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest, ECompareOperation Op)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(img_thresh, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void ThresholdingVector::Compare(ImageBuffer& Source, ImageBuffer& Dest, float Value, ECompareOperation Op)
{
   CheckSameSize(Source, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(compare, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), Value);
}

void ThresholdingVector::Compare(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest, ECompareOperation Op)
{
   CheckSameSize(Source1, Source2);
   CheckSameSize(Source1, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(img_compare, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

std::string SelectName(const char * name, ThresholdingVector::ECompareOperation Op)
{
   std::string str = name;

   switch (Op)
   {
   case ThresholdingVector::LT:
      str += "_LT";
      break;
   case ThresholdingVector::LQ:
      str += "_LQ";
      break;
   case ThresholdingVector::EQ:
      str += "_EQ";
      break;
   case ThresholdingVector::GQ:
      str += "_GQ";
      break;
   case ThresholdingVector::GT:
      str += "_GT";
      break;
   }

   return str;
}

}
