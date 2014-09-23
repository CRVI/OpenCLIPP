////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityFFT.h
//! @date   : Feb 2014
//!
//! @brief  : Image comparisons for pattern matching accelerated using FFT
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
#include "Statistics.h"
#include "Integral.h"
#include "Transform.h"

namespace OpenCLIPP
{

/// A program that does Pattern Matching
class CL_API ImageProximityFFT : public ImageProgram
{
public:

   ImageProximityFFT(COpenCL& CL)
   :  ImageProgram(CL, "ImageProximityFFT.cl"),
      m_integral(CL),
      m_statistics(CL),
      m_transform(CL),
      m_fft(CL)
   { }

   // If Template is small (<16x16 pixels), ImageProximity may be faster

   // FFT operations do not work on images bigger than 16.7Mpixels

   ///< Allocate internal temporary buffer and build the program
   void PrepareFor(ImageBase& Source, Image& Template);  

   /// cross-correlation template matching
   void CrossCorr(Image& Source, Image& Template, Image& Dest);

   /// cross-correlation template matching
   void CrossCorr_Norm(Image& Source, Image& Template, Image& Dest);

   /// square different template matching
   void SqrDistance(Image& Source, Image& Template, Image& Dest);

   /// Normalized square different template matching
   void SqrDistance_Norm(Image& Source, Image& Template, Image& Dest);

protected:

   /// Fourier spectrums of Template 
   std::shared_ptr<TempImage> m_templ_spect;

   /// Fourier spectrums of Source 
   std::shared_ptr<TempImage> m_source_spect;

   /// Fourier spectrums of result 
   std::shared_ptr<TempImage> m_result_spect;

   /// Square integral sum of source image
   std::shared_ptr<TempImage> m_image_sqsums;

   /// Template stored in a bigger image
   std::shared_ptr<TempImage> m_bigger_template;

   /// Source stored in a bigger image
   std::shared_ptr<TempImage> m_bigger_source;


   Integral   m_integral;
   Statistics m_statistics;
   Transform  m_transform;
   FFT        m_fft;

   /// prepare for the square different template matching
   void MatchSquareDiff(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest);

   /// prepare for the normalized square different template matching
   void MatchSquareDiffNorm(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest);

   /// prepare for the normalized cross correlation template matching
   void MatchCrossCorrNorm(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest);

   /// Performs a per-element multiplication of two Fourier spectrums and scales the result
   void MulAndScaleSpectrums(Image& Source, Image& Template, Image& Dest, float scale);

   /// Computes a convolution of two images
   void Convolve(Image& Source, Image& Template, Image& Dest);

   ImageProximityFFT& operator = (ImageProximityFFT&);   // Not a copyable object
};

}
