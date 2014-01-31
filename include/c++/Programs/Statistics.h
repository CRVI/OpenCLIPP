////////////////////////////////////////////////////////////////////////////////
//! @file	: Statistics.h
//! @date   : Jul 2013
//!
//! @brief  : Statistical reductions on images
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

/// A program that does statistical reductions
class CL_API Statistics : public ImageProgram
{
public:
   Statistics(COpenCL& CL)
   :  ImageProgram(CL, "Statistics.cl"),
      m_ResultBuffer(*m_CL, &m_Result, 1)
   { }

   double Min(IImage& Source);          ///< Finds the minimum value in the image
   double Max(IImage& Source);          ///< Finds the maximum value in the image
   double MinAbs(IImage& Source);       ///< Finds the minimum of the absolute of the values in the image
   double MaxAbs(IImage& Source);       ///< Finds the maxumum of the absolute of the values in the image
   double Sum(IImage& Source);          ///< Calculates the sum of all pixel values
   uint   CountNonZero(IImage& Source); ///< Calculates the number of non zero pixels
   double Mean(IImage& Source);         ///< Calculates the mean value of all pixel values
   double MeanSqr(IImage& Source);      ///< Calculates the mean of the square of all pixel values

protected:
   float m_Result;
   Buffer m_ResultBuffer;

   void PrepareBuffer(const ImageBase& Image);

   std::vector<float> m_PartialResult;
   std::shared_ptr<Buffer> m_PartialResultBuffer;

   void Init(IImage& Source);
   void InitAbs(IImage& Source);

   Statistics& operator = (Statistics&);   // Not a copyable object
};

}
