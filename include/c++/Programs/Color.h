////////////////////////////////////////////////////////////////////////////////
//! @file	: Color.h
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

#pragma once

#include "Program.h"


namespace OpenCLIPP
{

/// A program that converts 3 channel images to/from 4 channel.
/// It is used internally by the ColorImage class
class Color : public ImageBufferProgram
{
public:
   Color(COpenCL& CL)
   :  ImageBufferProgram(CL, "Color.cl")
   { }

   void Convert3CTo4C(ImageBuffer& Source, TempImage& Dest);
   void Convert4CTo3C(TempImage& Source, ImageBuffer& Dest);
};

}
