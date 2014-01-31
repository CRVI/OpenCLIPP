////////////////////////////////////////////////////////////////////////////////
//! @file	: Statistics.cpp
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

#include "Programs/Statistics.h"


#define KERNEL_RANGE(src_img) GetRange(src_img), GetLocalRange()
#define SELECT_NAME(name, src_img) SelectName( #name , src_img)

#include "kernel_helpers.h"

#include "StatisticsHelpers.h"


using namespace std;

namespace OpenCLIPP
{


// Statistics
void Statistics::PrepareBuffer(const ImageBase& Image)
{
   size_t NbGroups = (size_t) GetNbGroups(Image);

   // We need twice the size to be able to store the number of pixels per group
   size_t BufferSize = NbGroups * 2;

   if (m_PartialResultBuffer != nullptr &&
      m_PartialResultBuffer->Size() == BufferSize * sizeof(float) &&
      m_PartialResult.size() == BufferSize)
   {
      return;
   }

   m_PartialResult.assign(BufferSize, 0);

   m_PartialResultBuffer.reset();
   m_PartialResultBuffer = make_shared<Buffer>(*m_CL, m_PartialResult.data(), BufferSize);
}

// Init
void Statistics::Init(IImage& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Image2D, cl::Buffer>(SelectProgram(Source), "init")
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}

void Statistics::InitAbs(IImage& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Image2D, cl::Buffer>(SelectProgram(Source), "init_abs")
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}


// Reductions
double Statistics::Min(IImage& Source)
{
   Init(Source);

   Kernel(reduce_min, In(Source), Out(m_ResultBuffer), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double Statistics::Max(IImage& Source)
{
   Init(Source);

   Kernel(reduce_max, In(Source), Out(m_ResultBuffer), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double Statistics::MinAbs(IImage& Source)
{
   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(m_ResultBuffer), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double Statistics::MaxAbs(IImage& Source)
{
   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(m_ResultBuffer), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double Statistics::Sum(IImage& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(*m_PartialResultBuffer), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceSum(m_PartialResult);
}

uint Statistics::CountNonZero(IImage& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_count_nz, In(Source), Out(*m_PartialResultBuffer), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return (uint) ReduceSum(m_PartialResult);
}

double Statistics::Mean(IImage& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(*m_PartialResultBuffer), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

double Statistics::MeanSqr(IImage& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(*m_PartialResultBuffer), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

}
