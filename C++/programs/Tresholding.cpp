////////////////////////////////////////////////////////////////////////////////
//! @file	: Tresholding.cpp
//! @date   : Jul 2013
//!
//! @brief  : Image tresholding
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

#include "Programs/Tresholding.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void Tresholding::TresholdGT(IImage& Source, IImage& Dest, float Tresh, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(tresholdGT, In(Source), Out(Dest), Tresh, valueHigher);
}

void Tresholding::TresholdLT(IImage& Source, IImage& Dest, float Tresh, float valueLower)
{
   CheckCompatibility(Source, Dest);

   Kernel(tresholdLT, In(Source), Out(Dest), Tresh, valueLower);
}

void Tresholding::TresholdGTLT(IImage& Source, IImage& Dest, float threshLT, float valueLower, float treshGT, float valueHigher)
{
   CheckCompatibility(Source, Dest);

   Kernel(tresholdGTLT, In(Source), Out(Dest), threshLT, valueLower, treshGT, valueHigher);
}

#undef SELECT_NAME
#define SELECT_NAME(name, src_img) SelectName( #name , Op)

std::string SelectName(const char * name, Tresholding::ECompareOperation Op);

void Tresholding::treshold(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(img_tresh, In(Source1, Source2), Out(Dest));
}

void Tresholding::Compare(IImage& Source, IImage& Dest, float Value, ECompareOperation Op)
{
   CheckCompatibility(Source, Dest);

   Kernel(compare, In(Source), Out(Dest), Value);
}

void Tresholding::Compare(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(img_compare, In(Source1, Source2), Out(Dest));
}

std::string SelectName(const char * name, Tresholding::ECompareOperation Op)
{
   std::string str = name;

   switch (Op)
   {
   case Tresholding::LT:
      str += "_LT";
      break;
   case Tresholding::LQ:
      str += "_LQ";
      break;
   case Tresholding::EQ:
      str += "_EQ";
      break;
   case Tresholding::GQ:
      str += "_GQ";
      break;
   case Tresholding::GT:
      str += "_GT";
      break;
   }

   return str;
}

}
