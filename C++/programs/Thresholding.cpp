////////////////////////////////////////////////////////////////////////////////
//! @file	: Thresholding.cpp
//! @date   : Jul 2013
//!
//! @brief  : Image thresholding
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

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void Thresholding::ThresholdGT(IImage& Source, IImage& Dest, float Thresh, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(thresholdGT, In(Source), Out(Dest), Thresh, valueHigher);
}

void Thresholding::ThresholdLT(IImage& Source, IImage& Dest, float Thresh, float valueLower)
{
   CheckCompatibility(Source, Dest);

   Kernel(thresholdLT, In(Source), Out(Dest), Thresh, valueLower);
}

void Thresholding::ThresholdGTLT(IImage& Source, IImage& Dest, float threshLT, float valueLower, float threshGT, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(thresholdGTLT, In(Source), Out(Dest), threshLT, valueLower, threshGT, valueHigher);
}

#undef SELECT_NAME
#define SELECT_NAME(name, src_img) SelectName( #name , Op)

std::string SelectName(const char * name, Thresholding::ECompareOperation Op);

void Thresholding::Threshold(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(img_thresh, In(Source1, Source2), Out(Dest));
}

void Thresholding::Compare(IImage& Source, IImage& Dest, float Value, ECompareOperation Op)
{
   CheckSameSize(Source, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(compare, In(Source), Out(Dest), Value);
}

void Thresholding::Compare(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op)
{
   CheckSameSize(Source1, Source2);
   CheckSameSize(Source1, Dest);

   if (!Dest.IsUnsigned() || Dest.Depth() != 8 || Dest.NbChannels() != 1)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of Compare() must be unsigned 8 bit 1 channel");

   Kernel(img_compare, In(Source1, Source2), Out(Dest));
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
