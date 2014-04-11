////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.h
//! @date   : Jul 2013
//!
//! @brief  : Simple image transformation
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


/// A program that does transformations
class CL_API Transform : public ImageProgram
{
public:
   Transform(COpenCL& CL)
      :  ImageProgram(CL, "Transform.cl")
   { }

   /// Mirrors the image along X.
   /// D(x,y) = D(width - x - 1, y)
   void MirrorX(IImage& Source, IImage& Dest);

   /// Mirrors the image along Y.
   /// D(x,y) = D(x, height - y - 1)
   void MirrorY(IImage& Source, IImage& Dest);

   /// Flip : Mirrors the image along X and Y.
   /// D(x,y) = D(width - x - 1, height - y - 1)
   void Flip(IImage& Source, IImage& Dest);

   /// Transposes the image.
   /// Dest must have a width >= as Source's height and a height >= as Source's width
   /// D(x,y) = D(y, x)
   void Transpose(IImage& Source, IImage& Dest);

   /// Rotates the source image aroud the origin (0,0) and then shifts it.
   void Rotate(IImage& Source, IImage& Dest,
      double Angle, double XShift, double YShift, bool LinearInterpolation = true);

   /// Resizes the image.
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param LinearInterpolation : If true, linear interpolation will be used when stretching images.
   ///      If false, the nearest pixel value will be used.
   ///      Linear interpolation can be used only with F32 images.
   /// \param KeepRatio : If false, Dest will be filled with the image from source, potentially changing
   ///      the aspect ratio of the image. If true, the aspect ratio of the image will be kept, potentially
   ///      leaving part of Dest with invalid (unchaged) data to the right or to the bottom.
   void Resize(IImage& Source, IImage& Dest, bool LinearInterpolation = true, bool KeepRatio = false);

   /// Sets all values of Dest to value
   void SetAll(IImage& Dest, float Value);
};

}
