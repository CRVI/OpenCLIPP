////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.h
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformations
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
class CL_API Transform : public ImageProgram
{
public:
   Transform(COpenCL& CL)
   :  ImageProgram(CL, "Transform.cl")
   { }

   /// Lists the possible interpolation types useable in some primitives
   enum EInterpolationType
   {
      NearestNeighbour,   ///< Chooses the value of the closest pixel - Fastest
      Linear,             ///< Does a bilinear interpolation of the 4 closest pixels
      Cubic,              ///< Does a bicubic interpolation of the 16 closest pixels
      Lanczos2,           ///< Does 2-lobed Lanczos interpolation using 16 pixels
      Lanczos3,           ///< Does 3-lobed Lanczos interpolation using 36 pixels
      SuperSampling,      ///< Samples each pixel of the source for best resize result - for downsizing images only
      BestQuality,        ///< Automatically selects the choice that will give the best image quality for the operation
   };

   /// Mirrors the image along X.
   /// D(x,y) = D(width - x - 1, y)
   void MirrorX(Image& Source, Image& Dest);

   /// Mirrors the image along Y.
   /// D(x,y) = D(x, height - y - 1)
   void MirrorY(Image& Source, Image& Dest);

   /// Flip : Mirrors the image along X and Y.
   /// D(x,y) = D(width - x - 1, height - y - 1)
   void Flip(Image& Source, Image& Dest);

   /// Transposes the image.
   /// Dest must have a width >= as Source's height and a height >= as Source's width
   /// D(x,y) = D(y, x)
   void Transpose(Image& Source, Image& Dest);

   /// Rotates the source image aroud the origin (0,0) and then shifts it.
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param Angle : Angle to use for the rotation, in degrees.
   /// \param XShift : Shift along horizonltal axis to do after the rotation.
   /// \param YShift : Shift along vertical axis to do after the rotation.
   /// \param Interpolation : Type of interpolation to use.
   ///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   ///      BestQuality will use Cubic.
   void Rotate(Image& Source, Image& Dest,
      double Angle, double XShift, double YShift, EInterpolationType Interpolation = BestQuality);

   /// Resizes the image.
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param Interpolation : Type of interpolation to use.
   ///         Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   /// \param KeepRatio : If false, Dest will be filled with the image from source, potentially changing
   ///      the aspect ratio of the image. If true, the aspect ratio of the image will be kept, potentially
   ///      leaving part of Dest with invalid (unchaged) data to the right or to the bottom.
   void Resize(Image& Source, Image& Dest, EInterpolationType Interpolation = BestQuality, bool KeepRatio = false);

   /// Shearing transformation.
   /// \param Source : Source image
   /// \param Dest : Destination image
   /// \param ShearX : X Shearing coefficient.
   /// \param ShearY : Y Shearing coefficient.
   /// \param XShift : Shift along horizonltal axis to do after the shearing.
   /// \param YShift : Shift along vertical axis to do after the shearing.
   /// \param Interpolation : Type of interpolation to use.
   ///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   ///      BestQuality will use Cubic.
   void Shear(Image& Source, Image& Dest,
      double ShearX, double ShearY, double XShift, double YShift, EInterpolationType Interpolation = BestQuality);

   /// Remap
   /// \param Source : Source image
   /// \param MapX : X Map image, must be 1 channel, F32
   /// \param MapY : Y Map image, must be 1 channel, F32
   /// \param Dest : Destination image
   /// \param Interpolation : Type of interpolation to use.
   ///      Available choices are : NearestNeighbour, Linear, Cubic or BestQuality
   ///      BestQuality will use Cubic.
   void Remap(Image& Source, Image& MapX, Image& MapY, Image& Dest, EInterpolationType Interpolation = BestQuality);

   /// Sets all values of Dest to value
   void SetAll(Image& Dest, float Value);

   /// Sets all values inside destination rectangle of Dest to Value
   void SetAll(Image& Dest, uint X, uint Y, uint Width, uint Height, float Value);

protected:
   void ResizeLanczos(Image& Source, Image& Dest, int a, cl::NDRange Range);
};

}
