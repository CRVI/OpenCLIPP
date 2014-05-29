////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityFFT.h
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching woth FFT on image buffers
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
#include "FFT.h"
#include "StatisticsVector.h"
#include "IntegralBuffer.h"
#include "TransformBuffer.h"

namespace OpenCLIPP
{

	/// A program that does Pattern Matching
class CL_API ImageProximityFFT : public ImageBufferProgram
{
public:

   ImageProximityFFT(COpenCL& CL)
   :  ImageBufferProgram(CL, "ImageProximityFFT.cl"),
      m_integral(CL),
      m_statistics(CL),
      m_transform(CL),
      m_fft(CL)
   { }

   // If Template is small (<16x16 pixels), ImageProximityBuffer may be faster

   // FFT operations do not work on images bigger than 16.7Mpixels

   ///< Allocate internal temporary buffer and build the program
   void PrepareFor(ImageBase& Source, ImageBuffer& Template);  

   /// cross-correlation template matching
   void CrossCorr(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest);

   /// cross-correlation template matching
   void CrossCorr_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest);

   /// square different template matching
   void SqrDistance(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest);

   /// Normalized square different template matching
   void SqrDistance_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest);

protected:

   //Buffer of Template Fourier spectrums
   std::shared_ptr<TempImageBuffer> m_templ_spect;

   //Buffer of Source Fourier spectrums
   std::shared_ptr<TempImageBuffer> m_source_spect;

   //Buffer of result Fourier spectrums
   std::shared_ptr<TempImageBuffer> m_result_spect;

   //Buffer of square integral sum image
   std::shared_ptr<TempImageBuffer> m_image_sqsums;

   //Buffer of bigger template used for FFT
   std::shared_ptr<TempImageBuffer> m_bigger_template;

   //Buffer of bigger source used for FFT
   std::shared_ptr<TempImageBuffer> m_bigger_source;

   //Buffer of bigger dest used for FFT
   std::shared_ptr<TempImageBuffer> m_bigger_dest;

   //Buffer of source in float
   std::shared_ptr<TempImageBuffer> m_float_source;

   //Buffer of template in float
   std::shared_ptr<TempImageBuffer> m_float_template;

   IntegralBuffer   m_integral;
   StatisticsVector m_statistics;
   TransformBuffer  m_transform;
   FFT              m_fft;

   /// prepare for the square different template matching
   void MatchTemplatePrepared_SQDIFF(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest);

   /// prepare for the normalized square different template matching
   void MatchTemplatePrepared_SQDIFF_NORM(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest);

   /// prepare for the normalized cross correlation template matching
   void MatchTemplatePrepared_CCORR_NORM(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest);

   /// Performs a per-element multiplication of two Fourier spectrums and scales the result
   void MulAndScaleSpectrums(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest, float scale);

   /// Computes a convolution of two images
   void Convolve(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest);

   ImageProximityFFT& operator = (ImageProximityFFT&);   // Not a copyable object
};
}