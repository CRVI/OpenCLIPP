////////////////////////////////////////////////////////////////////////////////
//! @file	: Statistics.h
//! @date   : Jul 2013
//!
//! @brief  : Statistical reductions
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
      m_ResultBuffer(*m_CL, m_Result, 4)
   { }

   // These are for 1 channel images
   double Min(    Image& Source);                  ///< Finds the minimum value in the image
   double Max(    Image& Source);                  ///< Finds the maximum value in the image
   double MinAbs( Image& Source);                  ///< Finds the minimum of the absolute of the values in the image
   double MaxAbs( Image& Source);                  ///< Finds the maxumum of the absolute of the values in the image
   double Sum(    Image& Source);                  ///< Calculates the sum of all pixel values
   double SumSqr( Image& Source);                  ///< Calculates the sum of the square of all pixel values
   double Mean(   Image& Source);                  ///< Calculates the mean value of all pixel values
   double MeanSqr(Image& Source);                  ///< Calculates the mean of the square of all pixel values
   double StdDev( Image& Source);                  ///< Calculates the standard deviation of all pixel values
   double StdDev( Image& Source, double& outmean); ///< Calculates the standard deviation and mean of all pixel values
   uint   CountNonZero(Image& Source);             ///< Calculates the number of non zero pixels

   double Min(    Image& Source, int& outX, int& outY);  ///< Finds the position in the image that has the minimum value
   double Max(    Image& Source, int& outX, int& outY);  ///< Finds the position in the image that has the maximum value
   double MinAbs( Image& Source, int& outX, int& outY);  ///< Finds the position in the image that has the minimum of the absolute values
   double MaxAbs( Image& Source, int& outX, int& outY);  ///< Finds the position in the image that has the maximum of the absolute values

   // These are for 2, 3 or 4 channel images
   void Min(      Image& Source, double outVal[4]);   ///< Finds the minimum values in the image
   void Max(      Image& Source, double outVal[4]);   ///< Finds the maximum values in the image
   void MinAbs(   Image& Source, double outVal[4]);   ///< Finds the minimum of the absolute of the values in the image
   void MaxAbs(   Image& Source, double outVal[4]);   ///< Finds the maxumum of the absolute of the values in the image
   void Sum(      Image& Source, double outVal[4]);   ///< Calculates the sum of all pixel values
   void SumSqr(   Image& Source, double outVal[4]);   ///< Calculates the sum of the square of all pixel values
   void Mean(     Image& Source, double outVal[4]);   ///< Calculates the mean value of all pixel values
   void MeanSqr(  Image& Source, double outVal[4]);   ///< Calculates the mean of the square of all pixel values
   void StdDev(   Image& Source, double outVal[4]);   ///< Calculates the standard deviation of all pixel values
   void StdDev(   Image& Source, double outVal[4], double outMean[4]);  ///< Calculates the standard deviation and mean of all pixel values

protected:
   float m_Result[4];
   Buffer m_ResultBuffer;

   void PrepareBuffer(const ImageBase& Source);  // Prepares m_PartialResult
   void PrepareCoords(const ImageBase& Source);  // Prepares m_PartialResult and m_PartialCoord

   std::vector<float> m_PartialResult;    // CPU buffer that contains the partially calculated result
   std::vector<int>   m_PartialCoord;     // CPU buffer that contains the coordinates for the partially calculated result
   std::shared_ptr<Buffer> m_PartialResultBuffer;  // GPU buffer for m_PartialResult
   std::shared_ptr<Buffer> m_PartialCoordBuffer;   // GPU buffer for m_PartialCoord

   void Init(Image& Source);
   void InitAbs(Image& Source);

   Statistics& operator = (Statistics&);   // Not a copyable object
};

}
