////////////////////////////////////////////////////////////////////////////////
//! @file	: Color.cpp
//! @date   : Jul 2013
//!
//! @brief  : 3 Channel & 4 Channel image conversion
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

#include "Programs/Color.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void Color::Convert3CTo4C(ImageBuffer& Source, TempImage& Dest)
{
   CheckSizeAndType(Source, Dest);

   if (Source.NbChannels() != 3)
      throw cl::Error(CL_INVALID_VALUE, "Color::Convert3CTo4C needs an image with 3 channels as Source");

   if (Dest.NbChannels() != 4)
      throw cl::Error(CL_INVALID_VALUE, "Color::Convert3CTo4C needs an image with 4 channels as Dest");

   Kernel(Convert3CTo4C, Source, Dest, Source.Step());
}

void Color::Convert4CTo3C(TempImage& Source, ImageBuffer& Dest)
{
   CheckSizeAndType(Source, Dest);

   if (Source.NbChannels() != 4)
      throw cl::Error(CL_INVALID_VALUE, "Color::Convert4CTo3C needs an image with 4 channels as Source");

   if (Dest.NbChannels() != 3)
      throw cl::Error(CL_INVALID_VALUE, "Color::Convert4CTo3C needs an image with 3 channels as Dest");

   Kernel(Convert4CTo3C, Source, Dest, Dest.Step());
}

}
