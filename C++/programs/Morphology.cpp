////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.cpp
//! @date   : Jul 2013
//!
//! @brief  : Morphological operations on images
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

#include "kernel_helpers.h"


namespace OpenCLIPP
{

void Morphology::Erode(IImage& Source, IImage& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if ((Width & 1) == 0)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must be impair");

   if (Width < 3 || Width > 63)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must >= 3 && <= 63");

   if (Width == 3)
   {
      Kernel(erode3, Source, Dest);
   }
   else
   {
      Kernel(erode, Source, Dest, Width);
   }
   
}

void Morphology::Dilate(IImage& Source, IImage& Dest, int Width)
{
   CheckCompatibility(Source, Dest);

   if ((Width & 1) == 0)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must be impair");

   if (Width < 3 || Width > 63)
      throw cl::Error(CL_INVALID_ARG_VALUE, "Width for morphology operations must >= 3 && <= 63");

   if (Width == 3)
   {
      Kernel(dilate3, Source, Dest);
   }
   else
   {
      Kernel(dilate, Source, Dest, Width);
   }
   
}

void Morphology::Erode(IImage& Source, IImage& Dest, IImage& Temp, int Iterations, int Width)
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

void Morphology::Dilate(IImage& Source, IImage& Dest, IImage& Temp, int Iterations, int Width)
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

void Morphology::Open(IImage& Source, IImage& Dest, IImage& Temp, int Depth, int Width)
{
   Erode(Source, Temp, Dest, Depth, Width);
   Dilate(Temp, Dest, Temp, Depth, Width);
}

void Morphology::Close(IImage& Source, IImage& Dest, IImage& Temp, int Depth, int Width)
{
   Dilate(Source, Temp, Dest, Depth, Width);
   Erode(Temp, Dest, Temp, Depth, Width);
}

void Morphology::TopHat(IImage& Source, IImage& Dest, IImage& Temp, int Depth, int Width)
{
   Open(Source, Temp, Dest, Depth, Width);
   Kernel(sub_images, In(Source, Temp), Out(Dest));   // Source - Open
}

void Morphology::BlackHat(IImage& Source, IImage& Dest, IImage& Temp, int Depth, int Width)
{
   Close(Source, Temp, Dest, Depth, Width);
   Kernel(sub_images, In(Temp, Source), Out(Dest));   // Close - Source
}

void Morphology::Gradient(IImage& Source, IImage& Dest, IImage& Temp, int Width)
{
   Erode(Source, Temp, Width);
   Dilate(Source, Dest, Width);
   Kernel(sub_images, In(Dest, Temp), Out(Dest));     // Dilate - Erode
}

}
