////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.cpp
//! @date   : Jul 2013
//!
//! @brief  : Morphological operations
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

#include "Programs/Morphology.h"


#define KERNEL_RANGE(...)           GetRange(_FIRST(__VA_ARGS__), UseLocalRange(Width))
#define LOCAL_RANGE                 GetLocalRange(UseLocalRange(Width))
#define SELECT_NAME(name, src_img)  SelectName( #name , Width)

#include "kernel_helpers.h"

#include "WorkGroup.h"


namespace OpenCLIPP
{

static bool UseLocalRange(int /*Width*/)
{
   // We never use the local version - it seems to be slower than the simple version
   return false;
}

static std::string SelectName(const char * Name, int Width)  // Selects the proper kernel name
{
   std::string KernelName = Name;
   KernelName += std::to_string(Width);

   if (UseLocalRange(Width))
      KernelName += "_cached";

   return KernelName;
}


void Morphology::Erode(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if ((Width & 1) == 0)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must be impair");

   if (Width < 3 || Width > 63)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must >= 3 && <= 63");

   Kernel(erode, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Morphology::Dilate(Image& Source, Image& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if ((Width & 1) == 0)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must be impair");

   if (Width < 3 || Width > 63)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must >= 3 && <= 63");

   Kernel(dilate, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Morphology::Erode(Image& Source, Image& Dest, Image& Temp, int Iterations, int Width)
{
   if (Iterations <= 0)
      return;

   bool Pair = ((Iterations & 1) == 0);

   if (Pair)
   {
      Erode(Source, Temp, Width);
      Erode(Temp, Dest, Width);
   }
   else
      Erode(Source, Dest, Width);

   for (int i = 2; i < Iterations; i += 2)
   {
      Erode(Dest, Temp, Width);
      Erode(Temp, Dest, Width);
   }

}

void Morphology::Dilate(Image& Source, Image& Dest, Image& Temp, int Iterations, int Width)
{
   if (Iterations <= 0)
      return;

   bool Pair = ((Iterations & 1) == 0);

   if (Pair)
   {
      Dilate(Source, Temp, Width);
      Dilate(Temp, Dest, Width);
   }
   else
      Dilate(Source, Dest, Width);

   for (int i = 2; i < Iterations; i += 2)
   {
      Dilate(Dest, Temp, Width);
      Dilate(Temp, Dest, Width);
   }

}

void Morphology::Open(Image& Source, Image& Dest, Image& Temp, int Depth, int Width)
{
   Erode(Source, Temp, Dest, Depth, Width);
   Dilate(Temp, Dest, Temp, Depth, Width);
}

void Morphology::Close(Image& Source, Image& Dest, Image& Temp, int Depth, int Width)
{
   Dilate(Source, Temp, Dest, Depth, Width);
   Erode(Temp, Dest, Temp, Depth, Width);
}

void Morphology::TopHat(Image& Source, Image& Dest, Image& Temp, int Depth, int Width)
{
   Open(Source, Temp, Dest, Depth, Width);
   m_Arithmetic.Sub(Source, Temp, Dest);     // Source - Open
}

void Morphology::BlackHat(Image& Source, Image& Dest, Image& Temp, int Depth, int Width)
{
   Close(Source, Temp, Dest, Depth, Width);
   m_Arithmetic.Sub(Temp, Source, Dest);     // Close - Source
}

void Morphology::Gradient(Image& Source, Image& Dest, Image& Temp, int Width)
{
   Erode(Source, Temp, Width);
   Dilate(Source, Dest, Width);
   m_Arithmetic.Sub(Dest, Temp, Dest);     // Dilate - Erode
}

}
