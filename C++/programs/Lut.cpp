////////////////////////////////////////////////////////////////////////////////
//! @file	: Lut.cpp
//! @date   : Jul 2013
//!
//! @brief  : Lut transformation of images
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

#include "Programs/Lut.h"

#define VEC_WIDTH 4

#define KERNEL_RANGE(...) _FIRST(__VA_ARGS__).VectorRange(VEC_WIDTH)

#include "kernel_helpers.h"


namespace OpenCLIPP
{


void Lut::LUT(Image& Source, Image& Dest, uint * levels, uint * values, uint NbValues)
{
   CheckCompatibility(Source, Dest);

   if (SameType(Source, Dest) && Source.Depth() == 8 && Source.IsUnsigned() && Source.NbChannels() == 1)
   {
      // Use optimized version
      const static uint Length = 256;
      unsigned char Intensities[Length];
      uint level = 0;
      for (uint i = 0; i < Length; i++)
      {
         if (i < levels[0] || i >= levels[NbValues - 1])
            Intensities[i] = (unsigned char) i;
         else
         {
            while (level < NbValues - 1 && i >= levels[level + 1])
               level++;

            Intensities[i] = (unsigned char) values[level];
         }

      }

      BasicLut(Source, Dest, Intensities);
      
      return;
   }

   ReadBuffer Levels(*m_CL, levels, NbValues);
   ReadBuffer Values(*m_CL, values, NbValues);

   Kernel(LUT, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), Levels, Values, NbValues);
}

void Lut::LUTLinear(Image& Source, Image& Dest, float * levels, float * values, uint NbValues)
{
   CheckCompatibility(Source, Dest);

   ReadBuffer Levels(*m_CL, levels, NbValues);
   ReadBuffer Values(*m_CL, values, NbValues);

   Kernel(lut_linear, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), Levels, Values, NbValues);
}

void Lut::BasicLut(Image& Source, Image& Dest, unsigned char * values)
{
   if (Source.Depth() != 8 || !Source.IsUnsigned() || Source.NbChannels() != 1)
      throw cl::Error(CL_INVALID_VALUE, "BasicLut can only accept 1 channel unsigned integer images");

   CheckSizeAndType(Source, Dest);

   ReadBuffer Values(*m_CL, values, 256);

   if ((Source.Width() * Source.NbChannels() / VEC_WIDTH) % 16 || Source.Height() % 16)
   {
      // Standard version
      Kernel(lut_256, Source, Dest, Source.Step(), Dest.Step(),
         Source.Width() * Source.NbChannels(), Values);
   }
   else
   {
      // Faster version
      Kernel_(*m_CL, SelectProgram(Source), lut_256_cached, Source.VectorRange(VEC_WIDTH), cl::NDRange(16, 16, 1), Source, Dest, Source.Step(), Dest.Step(), Values)
   }

}

void Lut::Scale(Image& Source, Image& Dest, float SrcMin, float SrcMax, float DstMin, float DstMax)
{
   float Levels[2] = {SrcMin, SrcMax};
   float Values[2] = {DstMin, DstMax};

   LUTLinear(Source, Dest, Levels, Values, 2);
}

}
