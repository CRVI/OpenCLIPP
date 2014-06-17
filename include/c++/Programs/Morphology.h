////////////////////////////////////////////////////////////////////////////////
//! @file	: Morphology.h
//! @date   : Jul 2013
//!
//! @brief  : Morphological operations
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

#include "Arithmetic.h"

namespace OpenCLIPP
{

/// A program that does morphological operations
class CL_API Morphology : public ImageProgram
{
public:
   Morphology(COpenCL& CL)
   :  ImageProgram(CL, "Morphology.cl"),
      m_Arithmetic(CL)
   { }

   // 1 iteration
   void Erode(Image& Source, Image& Dest, int Width = 3);   ///< 1 Iteration
   void Dilate(Image& Source, Image& Dest, int Width = 3);  ///< 1 Iteration

   // Multiple iterations
   void Erode(Image& Source, Image& Dest, Image& Temp, int Iterations, int Width = 3);
   void Dilate(Image& Source, Image& Dest, Image& Temp, int Iterations, int Width = 3);

   void Open(Image& Source, Image& Dest, Image& Temp, int Depth = 1, int Width = 3);      ///< Erode then dilate
   void Close(Image& Source, Image& Dest, Image& Temp, int Depth = 1, int Width = 3);     ///< Dilate then erode
   void TopHat(Image& Source, Image& Dest, Image& Temp, int Depth = 1, int Width = 3);    ///< Source - Open
   void BlackHat(Image& Source, Image& Dest, Image& Temp, int Depth = 1, int Width = 3);  ///< Close - Source
   void Gradient(Image& Source, Image& Dest, Image& Temp, int Width = 3);                 ///< Dilate - Erode

private:
   Arithmetic m_Arithmetic;

};

}
