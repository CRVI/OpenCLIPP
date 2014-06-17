////////////////////////////////////////////////////////////////////////////////
//! @file	: Thresholding.cpp
//! @date   : Jul 2013
//!
//! @brief  : Thresholding operations
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

#include "Programs/Thresholding.h"


#define KERNEL_RANGE(...) GetRange(__VA_ARGS__)

#include "kernel_helpers.h"

namespace OpenCLIPP
{

std::string SelectName(const char * name, Thresholding::ECompareOperation Op);


void Thresholding::ThresholdGTLT(Image& Source, Image& Dest, float threshLT, float valueLower, float threshGT, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(thresholdGTLT, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), threshLT, valueLower, threshGT, valueHigher);
}


#undef SELECT_NAME
#define SELECT_NAME(name, src_img) SelectName( #name , Op)

void Thresholding::Threshold(Image& Source, Image& Dest, float Thresh, float value, ECompareOperation Op)
{
   CheckCompatibility(Source, Dest);

   Kernel(threshold, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), Thresh, value);
}

void Thresholding::Threshold(Image& Source1, Image& Source2, Image& Dest, ECompareOperation Op)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(img_thresh, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Thresholding::Compare(Image& Source, Image& Dest, float Value, ECompareOperation Op)
{
   CheckSameSize(Source, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(compare, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), Value);
}

void Thresholding::Compare(Image& Source1, Image& Source2, Image& Dest, ECompareOperation Op)
{
   CheckSameSize(Source1, Source2);
   CheckSameSize(Source1, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(img_compare, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

std::string SelectName(const char * name, Thresholding::ECompareOperation Op)
{
   std::string str = name;

   switch (Op)
   {
   case Thresholding::LT:
      str += "_LT";
      break;
   case Thresholding::LQ:
      str += "_LQ";
      break;
   case Thresholding::EQ:
      str += "_EQ";
      break;
   case Thresholding::GQ:
      str += "_GQ";
      break;
   case Thresholding::GT:
      str += "_GT";
      break;
   }

   return str;
}

}
