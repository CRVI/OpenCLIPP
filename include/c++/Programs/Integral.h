////////////////////////////////////////////////////////////////////////////////
//! @file	: Integral.h
//! @date   : Mar 2014
//!
//! @brief  : Calculates the square integral sum scan of an image
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
class CL_API Integral : public ImageProgram
{
public:
   Integral(COpenCL& CL)
      : ImageProgram(CL, "Integral.cl")
   { }

   /// Scans the image and generates the Integral sum into Dest - Dest must be F32 or F64
   void IntegralSum(Image& Source, Image& Dest);

   /// Scans the image and generates the Square Integral sum into Dest - Dest must be F32 or F64
   void SqrIntegral(Image& Source, Image& Dest);

   /// Allocates internal temporary buffers and builds the program
   void PrepareFor(ImageBase& Source);

protected:

   void Integral_F32(Image& Source, Image& Dest);
   void Integral_F64(Image& Source, Image& Dest);
   void SqrIntegral_F32(Image& Source, Image& Dest);
   void SqrIntegral_F64(Image& Source, Image& Dest);

   std::shared_ptr<TempImage> m_VerticalJunctions_F32;
   std::shared_ptr<TempImage> m_HorizontalJunctions_F32;
   std::shared_ptr<TempImage> m_VerticalJunctions_F64;
   std::shared_ptr<TempImage> m_HorizontalJunctions_F64;
};

}
