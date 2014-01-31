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

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void Lut::LUT(IImage& Source, IImage& Dest, uint * levels, uint * values, uint NbValues)
{
   CheckCompatibility(Source, Dest);

   ReadBuffer Levels(*m_CL, levels, NbValues);
   ReadBuffer Values(*m_CL, values, NbValues);

   if (Source.NbChannels() == 1)
   {
      Kernel(lut_1C, In(Source), Out(Dest), Levels, Values, NbValues);
      return;
   }

   Kernel(lut_4C, In(Source), Out(Dest), Levels, Values, NbValues);
}

void Lut::LUTLinear(IImage& Source, IImage& Dest, float * levels, float * values, uint NbValues)
{
   CheckCompatibility(Source, Dest);

   ReadBuffer Levels(*m_CL, levels, NbValues);
   ReadBuffer Values(*m_CL, values, NbValues);

   if (Source.NbChannels() == 1)
   {
      Kernel(lut_linear_1C, In(Source), Out(Dest), Levels, Values, NbValues);
      return;
   }

   Kernel(lut_linear_4C, In(Source), Out(Dest), Levels, Values, NbValues);
}

void Lut::Scale(IImage& Source, IImage& Dest, float SrcMin, float SrcMax, float DstMin, float DstMax)
{
   float Levels[2] = {SrcMin, SrcMax};
   float Values[2] = {DstMin, DstMax};

   LUTLinear(Source, Dest, Levels, Values, 2);
}

}
