////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityFFT.cpp
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on image buffers
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

#include "Programs/FFT.h"
#include "Programs/StatisticsVector.h"
#include "Programs/IntegralBuffer.h"
#include "Programs/TransformBuffer.h"
#include "Programs/ImageProximityFFT.h"
#include "kernel_helpers.h"

using namespace cl;

namespace OpenCLIPP
{
#ifdef USE_CLFFT

void ImageProximityFFT::PrepareFor(ImageBase& Source, ImageBuffer& Template)
{
   SSize size;

   size.Width = Source.Width();
   size.Height = Source.Height();

   if (m_image_sqsums == nullptr || m_image_sqsums->Width() < size.Width || m_image_sqsums->Height() < size.Height)
      m_image_sqsums = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 1);


   // Size of the FFT input and output
   size.Width = Source.Width() + Template.Width() / 2;
   size.Height = Source.Height() + Template.Height() / 2;

   // Search for a size supported by clFFT
   while (!m_fft.IsSupportedLength(size.Width))
      size.Width++;

   while (!m_fft.IsSupportedLength(size.Height))
      size.Height++;

   if (m_bigger_source == nullptr || m_bigger_source->Width() < size.Width || m_bigger_source->Height() < size.Height)
      m_bigger_source = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 1);
      
   if (m_bigger_template == nullptr || m_bigger_template->Width() < size.Width || m_bigger_template->Height() < size.Height)
      m_bigger_template = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 1);

   if (m_bigger_dest == nullptr || m_bigger_dest->Width() < size.Width || m_bigger_dest->Height() < size.Height)
      m_bigger_dest = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 1);


   // Size of the spectral images
   size.Width = size.Width / 2 + 1;

   if (m_templ_spect == nullptr || m_templ_spect->Width() < size.Width || m_templ_spect->Height() < size.Height)
      m_templ_spect = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 2);

   if (m_source_spect == nullptr || m_source_spect->Width() < size.Width || m_source_spect->Height() < size.Height)
      m_source_spect = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 2);

   if (m_result_spect == nullptr || m_result_spect->Width() < size.Width || m_result_spect->Height() < size.Height)
      m_result_spect = std::make_shared<TempImageBuffer>(*m_CL, size, SImage::F32, 2);

   m_integral.PrepareFor(Source);
   m_statistics.PrepareFor(Template);
   m_transform.PrepareFor(Source);
   m_fft.PrepareFor(*m_bigger_source, *m_source_spect);
   
}

void ImageProximityFFT::MatchTemplatePrepared_SQDIFF(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest)
{
   Check1Channel(Dest);
   Kernel(matchTemplatePreparedSQDIFF,  In(Source), Out(Dest), width, hight, templ_sqsum, Source.Step(), Dest.Step(), Dest.Width(), Dest.Height());
}

void ImageProximityFFT::MatchTemplatePrepared_SQDIFF_NORM(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest)
{
   Check1Channel(Dest);
   Kernel(matchTemplatePreparedSQDIFF_NORM,  In(Source), Out(Dest), width, hight, templ_sqsum, Source.Step(), Dest.Step(), Dest.Width(), Dest.Height());
}

void ImageProximityFFT::MatchTemplatePrepared_CCORR_NORM(int width, int hight, ImageBuffer& Source, float templ_sqsum, ImageBuffer& Dest)
{
   Check1Channel(Dest);
   Kernel(matchTemplatePreparedCCORR_NORM,  In(Source), Out(Dest), width, hight, templ_sqsum, Source.Step(), Dest.Step(), Dest.Width(), Dest.Height());
}

void ImageProximityFFT::CrossCorr(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   CheckSameSize(Source, Dest);
   Check1Channel(Source);
   Check1Channel(Template);
   CheckFloat(Dest);

   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   Convolve(Source, Template, Dest);
}

void ImageProximityFFT::CrossCorr_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   CheckSameSize(Source, Dest);
   Check1Channel(Source);
   Check1Channel(Template);
   CheckFloat(Dest);

   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   float templ_sqsum =  static_cast<float>(m_statistics.SumSqr(Template));

   Convolve(Source, Template, Dest);

   MatchTemplatePrepared_CCORR_NORM(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::SqrDistance(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   Check1Channel(Source);
   Check1Channel(Template);
   CheckFloat(Dest);

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   float templ_sqsum =  static_cast<float>(m_statistics.SumSqr(Template));

   CrossCorr(Source, Template, Dest);

   MatchTemplatePrepared_SQDIFF(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::SqrDistance_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   Check1Channel(Source);
   Check1Channel(Template);
   CheckFloat(Dest);

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   float templ_sqsum =  static_cast<float>(m_statistics.SumSqr(Template));

   CrossCorr(Source, Template, Dest);
   MatchTemplatePrepared_SQDIFF_NORM(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::MulAndScaleSpectrums(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest, float scale)
{
   CheckSameSize(Source, Template);
   CheckSameSize(Source, Dest);

   // Verify image types
   if (Source.DataType() != SImage::F32 || Template.DataType() != SImage::F32 || Dest.DataType() != SImage::F32)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The function works only with F32 images");

   // Verify image types
   if (Source.NbChannels() != 2 || Template.NbChannels() != 2 || Dest.NbChannels() != 2)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The function works only with 2 channel images");

   Kernel(mulAndScaleSpectrums, Source, Template, Dest, Source.Step(), Template.Step(), Dest.Step(), Source.Width(), Source.Height(), scale);
}

void ImageProximityFFT::Convolve(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   CheckSameSize(Source, Dest);

   PrepareFor(Source, Template);

   m_transform.SetAll(*m_bigger_template, 0);
   m_transform.SetAll(*m_bigger_source, 0);

   Kernel(copy_offset, In(Source), Out(*m_bigger_source), Source.Step(), m_bigger_source->Step(),
      int(Template.Width()) / 2, int(Template.Height()) / 2, m_bigger_source->Width(), m_bigger_source->Height());

   Kernel(copy_offset, In(Template), Out(*m_bigger_template), Template.Step(), m_bigger_template->Step(),
      0, 0, m_bigger_template->Width(), m_bigger_template->Height());

   m_fft.Forward(*m_bigger_source, *m_source_spect);
   m_fft.Forward(*m_bigger_template, *m_templ_spect);

   float Area = float(m_bigger_source->Width() * m_bigger_source->Height());

   MulAndScaleSpectrums(*m_source_spect, *m_templ_spect, *m_result_spect, 1 / Area );

   m_fft.Inverse(*m_result_spect, *m_bigger_dest);

   Kernel(copy_roi, In((*m_bigger_dest)), Out(Dest), m_bigger_dest->Step(), Dest.Step(), Dest.Width(), Dest.Height());
}

#else   // USE_CLFFT

void ImageProximityFFT::PrepareFor(ImageBase& , ImageBuffer& )
{ }

void ImageProximityFFT::MatchTemplatePrepared_SQDIFF(int , int , ImageBuffer& , float , ImageBuffer& )
{ }

void ImageProximityFFT::MatchTemplatePrepared_SQDIFF_NORM(int , int , ImageBuffer& , float , ImageBuffer& )
{ }

void ImageProximityFFT::MatchTemplatePrepared_CCORR_NORM(int , int , ImageBuffer& , float , ImageBuffer& )
{ }

void ImageProximityFFT::CrossCorr(ImageBuffer& , ImageBuffer& , ImageBuffer& )
{ }

void ImageProximityFFT::CrossCorr_Norm(ImageBuffer& , ImageBuffer& , ImageBuffer& )
{ }

void ImageProximityFFT::SqrDistance(ImageBuffer& , ImageBuffer& , ImageBuffer& )
{ }

void ImageProximityFFT::SqrDistance_Norm(ImageBuffer& , ImageBuffer& , ImageBuffer& )
{ }

void ImageProximityFFT::MulAndScaleSpectrums(ImageBuffer& , ImageBuffer& , ImageBuffer& , float )
{ }

void ImageProximityFFT::Convolve(ImageBuffer& , ImageBuffer& , ImageBuffer& )
{ }

#endif   // USE_CLFFT
}
