////////////////////////////////////////////////////////////////////////////////
//! @file	: Filters.h
//! @date   : Jul 2013
//!
//! @brief  : Convolution-type filters on images
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

/// A program for convolution-type filters on images
class CL_API Filters : public ImageProgram
{
public:
   Filters(COpenCL& CL)
   :  ImageProgram(CL, "Filters.cl")
   { }

   /// Gaussian blur filter.
   /// \param Sigma : Intensity of the filer - Allowed values : 0.01-10
   void GaussianBlur(IImage& Source, IImage& Dest, float Sigma);

   /// Gaussian filter.
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Gauss(IImage& Source, IImage& Dest, int Width);

   /// Sharpen filter.
   /// \param Width : Width of the filter box - Allowed values : 3
   void Sharpen(IImage& Source, IImage& Dest, int Width = 3);

   /// Smooth filter - or Box filter.
   /// \param Width : Width of the filter box - Allowed values : Impair & >=3
   void Smooth(IImage& Source, IImage& Dest, int Width = 3);

   /// Median filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Median(IImage& Source, IImage& Dest, int Width = 3);

   /// Vertical Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelVert(IImage& Source, IImage& Dest, int Width = 3);

   /// Horizontal Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelHoriz(IImage& Source, IImage& Dest, int Width = 3);

   /// Cross Sobel filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void SobelCross(IImage& Source, IImage& Dest, int Width = 3);

   /// Combined Sobel filter
   /// Does SobelVert & SobelHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Sobel(IImage& Source, IImage& Dest, int Width = 3);

   /// Vertical Prewitt filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void PrewittVert(IImage& Source, IImage& Dest, int Width = 3);

   /// Horizontal Prewitt filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void PrewittHoriz(IImage& Source, IImage& Dest, int Width = 3);

   /// Combined Prewitt filter
   /// Does PrewittVert & PrewittHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Prewitt(IImage& Source, IImage& Dest, int Width = 3);

   /// Vertical Scharr filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void ScharrVert(IImage& Source, IImage& Dest, int Width = 3);

   /// Horizontal Scharr filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void ScharrHoriz(IImage& Source, IImage& Dest, int Width = 3);

   /// Combined Scharr filter
   /// Does ScharrVert & ScharrHoriz and the combines the two with sqrt(V*V + H*H)
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Scharr(IImage& Source, IImage& Dest, int Width = 3);

   /// Hipass filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Hipass(IImage& Source, IImage& Dest, int Width = 3);

   /// Laplace filter
   /// \param Width : Width of the filter box - Allowed values : 3 or 5
   void Laplace(IImage& Source, IImage& Dest, int Width = 5);
};

}
