////////////////////////////////////////////////////////////////////////////////
//! @file	: FiltersVector.h
//! @date   : Jan 2014
//!
//! @brief  : Convolution-type filters on image buffers
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
class CL_API FiltersVector : public ImageBufferProgram
{
public:
   FiltersVector(COpenCL& CL)
   :  ImageBufferProgram(CL, "Vector_Filters.cl")
   { }

   /// Gaussian blur filter.
   /// \param Sigma : Intensity of the filer - Allowed values : 0.01-10
   void GaussianBlur(ImageBuffer& Source, ImageBuffer& Dest, float Sigma);

   /// Gaussian filter.
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Gauss(ImageBuffer& Source, ImageBuffer& Dest, int Width);

   /// Sharpen filter.
   /// \param Width : Width of the filter box - Allowed values : 3
   void Sharpen(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Smooth filter - or Box filter.
   /// \param Width : Width of the filter box - Allowed values : Impair & >=3
   void Smooth(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Median filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Median(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Vertical Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelVert(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Horizontal Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelHoriz(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Cross Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelCross(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Combined Sobel filter
   /// Does SobelVert & SobelHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Sobel(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Vertical Prewitt filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void PrewittVert(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Horizontal Prewitt filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void PrewittHoriz(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Combined Prewitt filter
   /// Does PrewittVert & PrewittHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Prewitt(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Vertical Scharr filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void ScharrVert(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Horizontal Scharr filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void ScharrHoriz(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Combined Scharr filter
   /// Does ScharrVert & ScharrHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Scharr(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Hipass filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Hipass(ImageBuffer& Source, ImageBuffer& Dest, int Width = 3);

   /// Laplace filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Laplace(ImageBuffer& Source, ImageBuffer& Dest, int Width = 5);
};

}
