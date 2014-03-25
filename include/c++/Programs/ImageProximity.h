////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.h
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on images
//! 
//! Copyright (C) 2014 - CRVI
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
	/// A program that does Pattern Matching
class CL_API ImageProximity : public ImageProgram
{
public:
   ImageProximity(COpenCL& CL)
   : ImageProgram(CL, "ImageProximity.cl")
   { }

   // Use only small template images (<=16x16 pixels)
   // Will be very slow if big template images are used
   void SqrDistance(IImage& Source, IImage& Template, IImage& Dest);
};

}
