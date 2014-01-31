////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.h
//! @date   : Jul 2013
//!
//! @brief  : Morphological operations on images
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

/// A program that does morphological operations
class CL_API Morphology : public ImageProgram
{
public:
   Morphology(COpenCL& CL)
   :  ImageProgram(CL, "Morphology.cl")
   { }

   // 1 iteration
   void Erode(IImage& Source, IImage& Dest, int Width = 3);   ///< 1 Iteration
   void Dilate(IImage& Source, IImage& Dest, int Width = 3);  ///< 1 Iteration

   // Multiple iterations
   void Erode(IImage& Source, IImage& Dest, IImage& Temp, int Iterations, int Width = 3);
   void Dilate(IImage& Source, IImage& Dest, IImage& Temp, int Iterations, int Width = 3);

   void Open(IImage& Source, IImage& Dest, IImage& Temp, int Depth = 1, int Width = 3);      ///< Erode then dilate
   void Close(IImage& Source, IImage& Dest, IImage& Temp, int Depth = 1, int Width = 3);     ///< Dilate then erode
   void TopHat(IImage& Source, IImage& Dest, IImage& Temp, int Depth = 1, int Width = 3);    ///< Source - Open
   void BlackHat(IImage& Source, IImage& Dest, IImage& Temp, int Depth = 1, int Width = 3);  ///< Close - Source
   void Gradient(IImage& Source, IImage& Dest, IImage& Temp, int Width = 3);                 ///< Dilate - Erode

};

}
