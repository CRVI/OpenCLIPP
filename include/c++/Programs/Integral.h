////////////////////////////////////////////////////////////////////////////////
//! @file	: Integral.h
//! @date   : Jul 2013
//!
//! @brief  : Calculates the integral sum scan of an image
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

/// A program that calculates the integral sum scan of an image
class CL_API Integral : public ImageProgram
{
public:
   Integral(COpenCL& CL)
      :  ImageProgram(CL, "Integral.cl")
   { }

   /// Scans the image and generates the Integral sum into Dest
   void IntegralSum(IImage& Source, IImage& Dest);

   /// Scans the image and generates the Square Integral sum into Dest
   void SqrIntegral(IImage& Source, IImage& Dest);

   /// Allocates internal temporary buffers and builds the program
   void PrepareFor(ImageBase& Source);

protected:
   std::shared_ptr<TempImage> m_VerticalJunctions;
   std::shared_ptr<TempImage> m_HorizontalJunctions;
};

}
