////////////////////////////////////////////////////////////////////////////////
//! @file	: ThresholdingVector.h
//! @date   : Mar 2014
//!
//! @brief  : Thresholding operations on image buffer
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

/// A program that does image thresholding
class CL_API ThresholdingVector : public VectorProgram
{
public:
   ThresholdingVector(COpenCL& CL)
   :  VectorProgram(CL, "Vector_Thresholding.cl")
   { }

   /// D = (S > Thresh ? valueHigher : S)
   void ThresholdGT(ImageBuffer& Source, ImageBuffer& Dest, float Thresh, float valueHigher = 255);

   /// D = (S < Thresh ? valueLower : S)
   void ThresholdLT(ImageBuffer& Source, ImageBuffer& Dest, float Thresh, float valueLower = 0);

   /// D = (S > Thresh ? valueHigher : (S < Thresh ? valueLower : S) )
   void ThresholdGTLT(ImageBuffer& Source, ImageBuffer& Dest, float threshLT, float valueLower, float threshGT, float valueHigher);

   enum ECompareOperation
   {
      LT,
      LQ,
      EQ,
      GQ,
      GT,
   };

   /// D = (S1 Op S2 ? S1 : S2)
   void Threshold(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest, ECompareOperation Op = GT);

   /// D = (S Op V)  - D will be 0 or 255
   /// Dest must be a 8U image
   void Compare(ImageBuffer& Source, ImageBuffer& Dest, float Value, ECompareOperation Op = GT);

   /// D = (S1 Op S2) - D will be 0 or 255
   /// Dest must be a 8U image
   void Compare(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest, ECompareOperation Op = GT);
};

}
