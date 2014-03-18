////////////////////////////////////////////////////////////////////////////////
//! @file	: StatisticsVector.h
//! @date   : Jul 2013
//!
//! @brief  : Statistical reductions on image buffers
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
class CL_API StatisticsVector : public ImageBufferProgram
{
public:
   StatisticsVector(COpenCL& CL)
   :  ImageBufferProgram(CL, "Vector_Statistics.cl"),
      m_ResultBuffer(*m_CL, &m_Result, 1)
   { }

   double Min(ImageBuffer& Source);           ///< Finds the minimum value in the image
   double Max(ImageBuffer& Source);           ///< Finds the maximum value in the image
   double MinAbs(ImageBuffer& Source);        ///< Finds the minimum of the absolute of the values in the image
   double MaxAbs(ImageBuffer& Source);        ///< Finds the maxumum of the absolute of the values in the image
   double Sum(ImageBuffer& Source);           ///< Calculates the sum of all pixel values
   uint   CountNonZero(ImageBuffer& Source);  ///< Calculates the number of non zero pixels
   double Mean(ImageBuffer& Source);          ///< Calculates the mean value of all pixel values
   double MeanSqr(ImageBuffer& Source);       ///< Calculates the mean of the square of all pixel values

   double Min(ImageBuffer& Source, int& outX, int& outY);     ///< Finds the position in the image that has the minimum value
   double Max(ImageBuffer& Source, int& outX, int& outY);     ///< Finds the position in the image that has the maximum value
   double MinAbs(ImageBuffer& Source, int& outX, int& outY);  ///< Finds the position in the image that has the minimum of the absolute values
   double MaxAbs(ImageBuffer& Source, int& outX, int& outY);  ///< Finds the position in the image that has the maximum of the absolute values

protected:
   float m_Result;
   Buffer m_ResultBuffer;

   void PrepareBuffer(const ImageBase& Image);  // Prepares m_PartialResult
   void PrepareCoords(const ImageBase& Image);  // Prepares m_PartialResult and m_PartialCoord

   std::vector<float> m_PartialResult;    // CPU buffer that contains the partially calculated result
   std::vector<int>   m_PartialCoord;     // CPU buffer that contains the coordinates for the partially calculated result
   std::shared_ptr<Buffer> m_PartialResultBuffer;  // GPU buffer for m_PartialResult
   std::shared_ptr<Buffer> m_PartialCoordBuffer;   // GPU buffer for m_PartialCoord

   void Init(ImageBuffer& Source);
   void InitAbs(ImageBuffer& Source);

   StatisticsVector& operator = (StatisticsVector&);   // Not a copyable object
};

}
