////////////////////////////////////////////////////////////////////////////////
//! @file	: Tresholding.h
//! @date   : Jul 2013
//!
//! @brief  : Image tresholding
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

/// A program that does image tresholding
class CL_API Tresholding : public ImageProgram
{
public:
   Tresholding(COpenCL& CL)
   :  ImageProgram(CL, "Tresholding.cl")
   { }

   /// D = (S > Tresh ? valueHigher : S)
   void TresholdGT(IImage& Source, IImage& Dest, float Tresh, float valueHigher = 255);

   /// D = (S < Tresh ? valueLower : S)
   void TresholdLT(IImage& Source, IImage& Dest, float Tresh, float valueLower = 0);

   /// D = (S > Tresh ? valueHigher : (S < Tresh ? valueLower : S) )
   void TresholdGTLT(IImage& Source, IImage& Dest, float threshLT, float valueLower, float treshGT, float valueHigher);

   enum ECompareOperation
   {
      LT,
      LQ,
      EQ,
      GQ,
      GT,
   };

   /// D = (S1 Op S2 ? S1 : S2)
   void treshold(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op = GT);

   /// D = (S Op V)  - D will be 0 or 1
   void Compare(IImage& Source, IImage& Dest, float Value, ECompareOperation Op = GT);

   /// D = (S1 Op S2) - D will be 0 or 1
   void Compare(IImage& Source1, IImage& Source2, IImage& Dest, ECompareOperation Op = GT);
};

}
