////////////////////////////////////////////////////////////////////////////////
//! @file	: FFT.cpp
//! @date   : Jan 2014
//!
//! @brief  : Fast Fourrier Transform using clFFT
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

// This file uses the open source library clFFT to efficiently compute 2D FFT
// clFFT is available here : https://github.com/clMathLibraries/clFFT
// Once you have the clFFT library built, define USE_CLFFT to use it
// If using Visual Studio, set CLFFT_DIR to the path of clFFT/build

#include "Programs/FFT.h"

#include "Programs/Program.h"


#ifdef USE_CLFFT

#include <clFFT.h>

#ifdef _MSC_VER
#pragma comment ( lib, "clFFT" )
#endif

#endif   // USE_CLFFT


using namespace std;

namespace OpenCLIPP
{

#ifdef USE_CLFFT

FFT::FFT(COpenCL& CL)
:  m_CL(&CL),
   m_Queue(CL),
   m_ForwardPlan(0),
   m_BackwardPlan(0)
{ }

FFT::~FFT()
{
   ReleasePlans();

   // Release clFFT library
   clfftTeardown();
}

void FFT::ReleasePlans()
{
   if (m_ForwardPlan == 0)
   {
      // Release the plans
      clfftDestroyPlan(&m_ForwardPlan);
      clfftDestroyPlan(&m_BackwardPlan);
      m_ForwardPlan = 0;
      m_BackwardPlan = 0;

      m_ForwardTempBuffer.reset();
      m_BackwardTempBuffer.reset();
   }

}

bool FFT::IsPlanCompatible(clfftPlanHandle ForwardPlan, const ImageBase& Real, const ImageBase& Complex)
{
   if (ForwardPlan == 0)
      return false;

   // Get values
   clfftPrecision Precision = ENDPRECISION;
   clfftGetPlanPrecision(ForwardPlan, &Precision);

   size_t Lengths[2] = {0};
   clfftGetPlanLength(ForwardPlan, CLFFT_2D, Lengths);

   size_t RealStrides[2] = {0};
   size_t ComplexStrides[2] = {0};
   clfftGetPlanInStride(ForwardPlan, CLFFT_2D, RealStrides);
   clfftGetPlanOutStride(ForwardPlan, CLFFT_2D, ComplexStrides);

   // Compare values
   if (Precision != CLFFT_SINGLE)
      return false;

   if (Lengths[0] != Real.Width())
      return false;

   if (Lengths[1] != Real.Height())
      return false;

   if (RealStrides[0] != 1 || RealStrides[1] != Real.ElementStep())
      return false;

   if (ComplexStrides[0] != 1 || ComplexStrides[1] * 2 != Complex.ElementStep())
      return false;

   return true;
}

void FFT::PrepareFor(const ImageBase& Real, const ImageBase& Complex)
{
   // Verify image types
   if (Real.DataType() != SImage::F32 || Complex.DataType() != SImage::F32)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "FFT works only with F32 images");

   if (Real.NbChannels() != 1)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "FFT works only with images with 1 channel for Real images");


   // Check if plans are ready and compatible with the images
   if (IsPlanCompatible(m_ForwardPlan, Real, Complex))
      return;  // Plan is good, we keep it as it is

   if (Complex.ElementStep() / 2 < Real.Width() / 2 + 1)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED,
         "FFT needs the complex image to be slightly larger than the real image to store all the data :"
         "Complex.ElementStep() / 2 must >= Real.Width() / 2 + 1");


   // Generate clFFT plans
   if (m_ForwardPlan == 0)
   {
      // First plan cleation - Setup clFFT first
      clfftSetupData fftSetup;
      cl_int err = clfftInitSetupData(&fftSetup);
      err = clfftSetup(&fftSetup);
   }
   else
      ReleasePlans();

   // TODO : Add error handling

   // Create a default plan for a forward FFT
   size_t clLengths[2] = {Real.Width(), Real.Height()};
   cl_int err = clfftCreateDefaultPlan(&m_ForwardPlan, *m_CL, CLFFT_2D, clLengths);

   // Set plan parameters
   err = clfftSetPlanPrecision(m_ForwardPlan, CLFFT_SINGLE);
   err = clfftSetLayout(m_ForwardPlan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
   err = clfftSetResultLocation(m_ForwardPlan, CLFFT_OUTOFPLACE);

   size_t RealStrides[2] = {1, Real.ElementStep()};
   size_t ComplexStrides[2] = {1, Complex.ElementStep() / 2};
   err = clfftSetPlanInStride(m_ForwardPlan, CLFFT_2D, RealStrides);
   err = clfftSetPlanOutStride(m_ForwardPlan, CLFFT_2D, ComplexStrides);
   err = clfftSetPlanScale(m_ForwardPlan, CLFFT_FORWARD, 1);

   // Bake the plan
   err = clfftBakePlan(m_ForwardPlan, 1, &m_Queue, NULL, NULL);

   // Create a default plan for a backward FFT
   err = clfftCreateDefaultPlan(&m_BackwardPlan, *m_CL, CLFFT_2D, clLengths);

   // Set plan parameters
   err = clfftSetPlanPrecision(m_BackwardPlan, CLFFT_SINGLE);
   err = clfftSetLayout(m_BackwardPlan, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
   err = clfftSetResultLocation(m_BackwardPlan, CLFFT_OUTOFPLACE);
   err = clfftSetPlanInStride(m_BackwardPlan, CLFFT_2D, ComplexStrides);
   err = clfftSetPlanOutStride(m_BackwardPlan, CLFFT_2D, RealStrides);
   err = clfftSetPlanScale(m_BackwardPlan, CLFFT_BACKWARD, 1);

   // Bake the plan
   err = clfftBakePlan(m_BackwardPlan, 1, &m_Queue, NULL, NULL);


   // Allocate temporary buffers
   size_t TempBufferSize = 0;
   err = clfftGetTmpBufSize(m_ForwardPlan, &TempBufferSize);

   if (TempBufferSize > 0)
      m_ForwardTempBuffer = make_shared<TempBuffer>(*m_CL, TempBufferSize);

   TempBufferSize = 0;
   err = clfftGetTmpBufSize(m_BackwardPlan, &TempBufferSize);

   if (TempBufferSize > 0)
      m_BackwardTempBuffer = make_shared<TempBuffer>(*m_CL, TempBufferSize);
}

void FFT::Forward(ImageBuffer& RealSource, ImageBuffer& ComplexDest)
{
   PrepareFor(RealSource, ComplexDest);

   cl_mem Input = RealSource, Output = ComplexDest;

   RealSource.SendIfNeeded();

   // Execute the plan
   cl_int err = clfftEnqueueTransform(m_ForwardPlan, CLFFT_FORWARD, 1, &m_Queue, 0, NULL, NULL,
      &Input, &Output, *m_ForwardTempBuffer);

   if (err != CL_SUCCESS)
      throw cl::Error(err, "FFT:: clfftEnqueueTransform");

   ComplexDest.SetInDevice();
}

void FFT::Inverse(ImageBuffer& ComplexSource, ImageBuffer& RealDest)
{
   PrepareFor(RealDest, ComplexSource);

   cl_mem Input = ComplexSource, Output = RealDest;

   ComplexSource.SendIfNeeded();

   // Execute the plan
   cl_int err = clfftEnqueueTransform(m_BackwardPlan, CLFFT_BACKWARD, 1, &m_Queue, 0, NULL, NULL,
      &Input, &Output, *m_BackwardTempBuffer);

   RealDest.Read(true);

   if (err != CL_SUCCESS)
      throw cl::Error(err, "FFT:: clfftEnqueueTransform");

   RealDest.SetInDevice();
}

#else   // USE_CLFFT

FFT::FFT(COpenCL&) { }

FFT::~FFT() { }

#endif   // USE_CLFFT

}
