////////////////////////////////////////////////////////////////////////////////
//! @file	: TransformBuffer.h
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformation on image buffers
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

/// A program that does transformations
class CL_API TransformBuffer : public ImageBufferProgram
{
public:
   TransformBuffer(COpenCL& CL)
   :  ImageBufferProgram(CL, "Transform_Buffer.cl")
   { }

   /// Mirrors the image along X.
   /// D(x,y) = D(width - x - 1, y)
   void MirrorX(ImageBuffer& Source, ImageBuffer& Dest);

   /// Mirrors the image along Y.
   /// D(x,y) = D(x, height - y - 1)
   void MirrorY(ImageBuffer& Source, ImageBuffer& Dest);

   /// Flip : Mirrors the image along X and Y.
   /// D(x,y) = D(width - x - 1, height - y - 1)
   void Flip(ImageBuffer& Source, ImageBuffer& Dest);

   /// Transposes the image.
   /// Dest must have a width >= as Source's height and a height >= as Source's width
   /// D(x,y) = D(y, x)
   void Transpose(ImageBuffer& Source, ImageBuffer& Dest);

   /// Rotates the source image aroud the origin (0,0) and then shifts it.
   void Rotate(ImageBuffer& Source, ImageBuffer& Dest,
      double Angle, double XShift, double YShift, bool LinearInterpolation = true);

   /// Sets all values of Dest to value
   void SetAll(ImageBuffer& Dest, float Value);
};

}
