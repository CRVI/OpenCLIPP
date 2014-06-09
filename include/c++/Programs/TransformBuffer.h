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

   /// Lists the possible interpolation types useable in some primitives
   enum EInterpolationType
   {
      NearestNeighbour,   ///< Chooses the value of the closest pixel - Fastest
      Linear,             ///< Does a bilinear interpolation of the 4 closest pixels
      Cubic,              ///< Unavailable
      SuperSampling,      ///< Unavailable
      BestQuality,        ///< Automatically selects the choice that will give the best image quality for the operation
   };

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
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param Angle : Angle to use for the rotation, in degrees.
   /// \param XShift : Shift along horizonltal axis to do after the rotation.
   /// \param YShift : Shift along vertical axis to do after the rotation.
   /// \param Interpolation : Type of interpolation to use.
   ///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   ///      BestQuality will use Cubic.
   void Rotate(ImageBuffer& Source, ImageBuffer& Dest,
      double Angle, double XShift, double YShift, EInterpolationType Interpolation = BestQuality);

   /// Resizes the image.
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param Interpolation : Type of interpolation to use.
   ///         Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   /// \param KeepRatio : If false, Dest will be filled with the image from source, potentially changing
   ///      the aspect ratio of the image. If true, the aspect ratio of the image will be kept, potentially
   ///      leaving part of Dest with invalid (unchaged) data to the right or to the bottom.
   void Resize(ImageBuffer& Source, ImageBuffer& Dest, EInterpolationType Interpolation = BestQuality, bool KeepRatio = false);

   /// Sets all values of Dest to value
   void SetAll(ImageBuffer& Dest, float Value);
};

}
