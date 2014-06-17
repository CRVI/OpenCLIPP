////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.h
//! @date   : Feb 2014
//!
//! @brief  : Image comparisons for pattern matching
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

/// A program for convolution-type filters on images
class CL_API ImageProximity : public ImageProgram
{
public:
   ImageProximity(COpenCL& CL)
   :  ImageProgram(CL, "ImageProximity.cl")
   { }

   // Use only small template images (<=16x16 pixels)
   // Will be very slow if big template images are used
   // For faster image proximity operations with big template image, use ImageProximityFFT

   //Computes the Euclidean distance between an image and a tamplate
   void SqrDistance(Image& Source, Image& Template, Image& Dest);

   //Computes the normalized Euclidean distance between an image and a tamplate
   void SqrDistance_Norm(Image& Source, Image& Template, Image& Dest);

   //Computes the sum of the absolute difference between an image and a tamplate
   void AbsDistance(Image& Source, Image& Template, Image& Dest);

   //Computes normalized cross-correlation between an image and a template.
   void CrossCorr(Image& Source, Image& Template, Image& Dest);

   //Computes normalized the cross-correlation between an image and a tamplate
   void CrossCorr_Norm(Image& Source, Image& Template, Image& Dest);  
};

}