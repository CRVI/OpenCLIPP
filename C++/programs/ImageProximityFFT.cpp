////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityFFT.cpp
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

#include "Programs/FFT.h"
#include "Programs/Statistics.h"
#include "Programs/Integral.h"
#include "Programs/Transform.h"
#include "Programs/ImageProximityFFT.h"
#include "kernel_helpers.h"

using namespace cl;

namespace OpenCLIPP
{
#ifdef USE_CLFFT

void ImageProximityFFT::PrepareFor(ImageBase& Source, Image& Template)
{
   SSize size;

   size.Width = Source.Width();
   size.Height = Source.Height();

   if (m_image_sqsums == nullptr || m_image_sqsums->Width() < size.Width || m_image_sqsums->Height() < size.Height)
      m_image_sqsums = std::make_shared<TempImage>(*m_CL, size, SImage::F32, Source.NbChannels());


   // Size of the FFT input and output
   size.Width = Source.Width() + Template.Width() / 2;
   size.Height = Source.Height() + Template.Height() / 2;

   // Search for a size supported by clFFT
   while (!m_fft.IsSupportedLength(size.Width))
      size.Width++;

   while (!m_fft.IsSupportedLength(size.Height))
      size.Height++;

   if (m_bigger_source == nullptr || m_bigger_source->Width() != size.Width || m_bigger_source->Height() != size.Height)
      m_bigger_source = std::make_shared<TempImage>(*m_CL, size, SImage::F32, 1);

   if (m_bigger_template == nullptr || m_bigger_template->Width() < size.Width || m_bigger_template->Height() < size.Height)
      m_bigger_template = std::make_shared<TempImage>(*m_CL, size, SImage::F32, 1);



   // Size of the spectral images
   size.Width = size.Width / 2 + 1;

   if (m_templ_spect == nullptr || m_templ_spect->Width() < size.Width || m_templ_spect->Height() < size.Height)
      m_templ_spect = std::make_shared<TempImage>(*m_CL, size, SImage::F32, 2);

   if (m_source_spect == nullptr || m_source_spect->Width() != size.Width || m_source_spect->Height() != size.Height)
      m_source_spect = std::make_shared<TempImage>(*m_CL, size, SImage::F32, 2);

   if (m_result_spect == nullptr || m_result_spect->Width() < size.Width || m_result_spect->Height() < size.Height)
      m_result_spect = std::make_shared<TempImage>(*m_CL, size, SImage::F32, 2);

   m_integral.PrepareFor(Source);
   m_statistics.PrepareFor(Template);
   m_transform.PrepareFor(Source);
   m_fft.PrepareFor(*m_bigger_source, *m_source_spect);
   SelectProgram(Source).Build();
   SelectProgram(*m_source_spect).Build();
}

void ImageProximityFFT::MatchSquareDiff(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest)
{
   cl_float4 TemplateSqSum = {float(templ_sqsum[0]), float(templ_sqsum[1]), float(templ_sqsum[2]), float(templ_sqsum[3])};

   Kernel(square_difference,        In(Source), Out(Dest), Source.Step(), Dest.Width(), Dest.Height(), Dest.Step(),
      width, hight, TemplateSqSum);
}

void ImageProximityFFT::MatchSquareDiffNorm(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest)
{
   cl_float4 TemplateSqSum = {float(templ_sqsum[0]), float(templ_sqsum[1]), float(templ_sqsum[2]), float(templ_sqsum[3])};

   Kernel(square_difference_norm,   In(Source), Out(Dest), Source.Step(), Dest.Step(), Dest.Width(), Dest.Height(),
      width, hight, TemplateSqSum);
}

void ImageProximityFFT::MatchCrossCorrNorm(int width, int hight, Image& Source, double * templ_sqsum, Image& Dest)
{
   cl_float4 TemplateSqSum = {float(templ_sqsum[0]), float(templ_sqsum[1]), float(templ_sqsum[2]), float(templ_sqsum[3])};

   Kernel(crosscorr_norm,           In(Source), Out(Dest), Source.Step(), Dest.Step(), Dest.Width(), Dest.Height(),
      width, hight, TemplateSqSum);
}

void ImageProximityFFT::CrossCorr(Image& Source, Image& Template, Image& Dest)
{
   CheckFloat(Dest);
   CheckSameNbChannels(Source, Template);
   CheckSameNbChannels(Source, Dest);
   CheckSameSize(Source, Dest);

   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   PrepareFor(Source, Template);

   Convolve(Source, Template, Dest);   // Computes the cross correlation using FFT
}

void ImageProximityFFT::CrossCorr_Norm(Image& Source, Image& Template, Image& Dest)
{
   CheckFloat(Dest);
   CheckSameNbChannels(Source, Template);
   CheckSameNbChannels(Source, Dest);
   CheckSameSize(Source, Dest);

   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   double templ_sqsum[4] = {0};
   m_statistics.SumSqr(Template, templ_sqsum);

   Convolve(Source, Template, Dest);   // Computes the cross correlation using FFT

   MatchCrossCorrNorm(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::SqrDistance(Image& Source, Image& Template, Image& Dest)
{
   CheckFloat(Dest);
   CheckSameNbChannels(Source, Template);
   CheckSameNbChannels(Source, Dest);
   CheckSameSize(Source, Dest);

   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   double templ_sqsum[4] = {0};
   m_statistics.SumSqr(Template, templ_sqsum);

   Convolve(Source, Template, Dest);   // Computes the cross correlation using FFT

   MatchSquareDiff(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::SqrDistance_Norm(Image& Source, Image& Template, Image& Dest)
{
   CheckFloat(Dest);
   CheckSameNbChannels(Source, Template);
   CheckSameNbChannels(Source, Dest);
   CheckSameSize(Source, Dest);

   // Verify image size
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "The template image must be smaller than source image.");

   // Verify image types
   if(!SameType(Source, Template))
      throw cl::Error(CL_IMAGE_FORMAT_MISMATCH, "The source image and the template image must be same type.");

   PrepareFor(Source, Template);

   m_integral.SqrIntegral(Source, *m_image_sqsums);

   double templ_sqsum[4] = {0};
   m_statistics.SumSqr(Template, templ_sqsum);

   CrossCorr(Source, Template, Dest);
   MatchSquareDiffNorm(Template.Width(), Template.Height(), *m_image_sqsums, templ_sqsum, Dest);
}

void ImageProximityFFT::MulAndScaleSpectrums(Image& Source, Image& Template, Image& Dest, float scale)
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

void ImageProximityFFT::Convolve(Image& Source, Image& Template, Image& Dest)
{
   CheckSameSize(Source, Dest);

   PrepareFor(Source, Template);

   // Fill these images with black
   m_transform.SetAll(*m_bigger_template, 0);
   m_transform.SetAll(*m_bigger_source, 0);

   for (uint i = 1; i <= Source.NbChannels(); i++)
   {
      // Copy the data from Source and Template in images that are F32 and are big enough
      Kernel(copy_offset, In(Source), Out(*m_bigger_source), Source.Step(), m_bigger_source->Step(),
         int(Template.Width()) / 2, int(Template.Height()) / 2, m_bigger_source->Width(), m_bigger_source->Height(), i);

      Kernel(copy_offset, In(Template), Out(*m_bigger_template), Template.Step(), m_bigger_template->Step(),
         0, 0, m_bigger_template->Width(), m_bigger_template->Height(), i);


      // Forward FFT of Source and Template
      m_fft.Forward(*m_bigger_source, *m_source_spect);
      m_fft.Forward(*m_bigger_template, *m_templ_spect);


      // We need to divide the values by the FFT area to get the proper 
      float Area = float(m_bigger_source->Width() * m_bigger_source->Height());

      // Do the convolution using pointwise product of the spectrums
      // See information here : http://en.wikipedia.org/wiki/Convolution_theorem
      MulAndScaleSpectrums(*m_source_spect, *m_templ_spect, *m_result_spect, 1 / Area);


      // Inverse FFT to get the result of the convolution
      m_fft.Inverse(*m_result_spect, *m_bigger_source);  // Reuse m_bigger_source image for the convolution result


      // Copy the result to Dest
      Kernel_(*m_CL, SelectProgram(Dest), copy_result, m_bigger_source->FullRange(), LOCAL_RANGE,
         In(*m_bigger_source), Out(Dest), m_bigger_source->Step(), Dest.Step(), Dest.Width(), Dest.Height(), i);
   }

}

#else   // USE_CLFFT

void ImageProximityFFT::PrepareFor(ImageBase& , Image& )
{ }

void ImageProximityFFT::MatchSquareDiff(int , int , Image& , double * , Image& )
{ }

void ImageProximityFFT::MatchSquareDiffNorm(int , int , Image& , double * , Image& )
{ }

void ImageProximityFFT::MatchCrossCorrNorm(int , int , Image& , double * , Image& )
{ }

void ImageProximityFFT::CrossCorr(Image& , Image& , Image& )
{ }

void ImageProximityFFT::CrossCorr_Norm(Image& , Image& , Image& )
{ }

void ImageProximityFFT::SqrDistance(Image& , Image& , Image& )
{ }

void ImageProximityFFT::SqrDistance_Norm(Image& , Image& , Image& )
{ }

void ImageProximityFFT::MulAndScaleSpectrums(Image& , Image& , Image& , float )
{ }

void ImageProximityFFT::Convolve(Image& , Image& , Image& )
{ }

#endif   // USE_CLFFT
}
