////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.h
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations on image buffers
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

/// A program that does Arithmetic operations on image buffers
class CL_API Arithmetic : public VectorProgram
{
public:
   Arithmetic(COpenCL& CL)
      :  VectorProgram(CL, "Arithmetic.cl")
   { }


   // Between two images
   void Add(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = S1 + S2
   void AddSquare(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);   ///< D = S1 + S2 * S2
   void Sub(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = S1 - S2
   void AbsDiff(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);     ///< D = abs(S1 - S2)
   void Mul(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = S1 * S2
   void Div(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = S1 / S2
   void Min(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = min(S1, S2)
   void Max(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);         ///< D = max(S1, S1)
   void Mean(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);        ///< D = (S1 + S2) / 2
   void Combine(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);     ///< D = sqrt(S1 * S1 + S2 * S2)

   // Image and value
   void Add(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = S + v
   void Sub(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = S - v
   void AbsDiff(ImageBuffer& Source, ImageBuffer& Dest, float value);   ///< D = abs(S - v)
   void Mul(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = S * v
   void Div(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = S / v
   void RevDiv(ImageBuffer& Source, ImageBuffer& Dest, float value);    ///< D = v / S
   void Min(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = min(S, v)
   void Max(ImageBuffer& Source, ImageBuffer& Dest, float value);       ///< D = max(S, v)
   void Mean(ImageBuffer& Source, ImageBuffer& Dest, float value);      ///< D = (S + V) / 2

   // Calculation on one image
   void Abs(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = abs(S)
   void Invert(ImageBuffer& Source, ImageBuffer& Dest);  ///< D = 255 - S
   void Sqr(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = S * S

   // Calculation on one image - float required
   void Exp(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = exp(S)
   void Log(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = log(S)
   void Sqrt(ImageBuffer& Source, ImageBuffer& Dest);    ///< D = sqrt(S)
   void Sin(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = sin(S)
   void Cos(ImageBuffer& Source, ImageBuffer& Dest);     ///< D = cos(S)

};

}
