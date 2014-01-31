////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.h
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations on images
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

/// A program that does Arithmetic operations on images
class CL_API Arithmetic : public ImageProgram
{
public:
   Arithmetic(COpenCL& CL)
   :  ImageProgram(CL, "Arithmetic.cl")
   { }


   // Between two images
   void Add(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = S1 + S2
   void AddSquare(IImage& Source1, IImage& Source2, IImage& Dest);   ///< D = S1 + S2 * S2
   void Sub(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = S1 - S2
   void AbsDiff(IImage& Source1, IImage& Source2, IImage& Dest);     ///< D = abs(S1 - S2)
   void Mul(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = S1 * S2
   void Div(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = S1 / S2
   void Min(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = min(S1, S2)
   void Max(IImage& Source1, IImage& Source2, IImage& Dest);         ///< D = max(S1, S1)
   void Mean(IImage& Source1, IImage& Source2, IImage& Dest);        ///< D = (S1 + S2) / 2
   void Combine(IImage& Source1, IImage& Source2, IImage& Dest);     ///< D = sqrt(S1 * S1 + S2 * S2)

   // Image and value
   void Add(IImage& Source, IImage& Dest, float value);       ///< D = S + v
   void Sub(IImage& Source, IImage& Dest, float value);       ///< D = S - v
   void AbsDiff(IImage& Source, IImage& Dest, float value);   ///< D = abs(S - v)
   void Mul(IImage& Source, IImage& Dest, float value);       ///< D = S * v
   void Div(IImage& Source, IImage& Dest, float value);       ///< D = S / v
   void RevDiv(IImage& Source, IImage& Dest, float value);    ///< D = v / S
   void Min(IImage& Source, IImage& Dest, float value);       ///< D = min(S, v)
   void Max(IImage& Source, IImage& Dest, float value);       ///< D = max(S, v)
   void Mean(IImage& Source, IImage& Dest, float value);      ///< D = (S + V) / 2

   // Calculation on one image
   void Abs(IImage& Source, IImage& Dest);     ///< D = abs(S)
   void Invert(IImage& Source, IImage& Dest);  ///< D = 255 - S
   void Exp(IImage& Source, IImage& Dest);     ///< D = exp(S)
   void Log(IImage& Source, IImage& Dest);     ///< D = log(S)
   void Sqr(IImage& Source, IImage& Dest);     ///< D = S * S
   void Sqrt(IImage& Source, IImage& Dest);    ///< D = sqrt(S)
   void Sin(IImage& Source, IImage& Dest);     ///< D = sin(S)
   void Cos(IImage& Source, IImage& Dest);     ///< D = cos(S)

};

}
