////////////////////////////////////////////////////////////////////////////////
//! @file	: StatisticsVector.cpp
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

#include "Programs/StatisticsVector.h"


#define KERNEL_RANGE(src_img) GetRange(src_img), GetLocalRange()
#define SELECT_NAME(name, src_img) SelectName( #name , src_img)

#include "kernel_helpers.h"

#include "StatisticsHelpers.h"


using namespace std;

namespace OpenCLIPP
{

// StatisticsVector
void StatisticsVector::PrepareBuffer(const ImageBase& Image)
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

void StatisticsVector::PrepareCoords(const ImageBase& Image)
{
   PrepareBuffer(Image);

   size_t NbGroups = (size_t) GetNbGroups(Image);

   // We are storing X and Y 
   size_t BufferSize = NbGroups * 2;

   if (m_PartialCoordBuffer != nullptr &&
      m_PartialCoordBuffer->Size() == BufferSize * sizeof(int) &&
      m_PartialCoord.size() == BufferSize)
   {
      return;
   }

   m_PartialCoord.assign(BufferSize, 0);

   m_PartialCoordBuffer.reset();
   m_PartialCoordBuffer = make_shared<Buffer>(*m_CL, m_PartialCoord.data(), BufferSize);
}


// Init
void StatisticsVector::Init(ImageBuffer& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Buffer, cl::Buffer>(SelectProgram(Source), "init")
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}

void StatisticsVector::InitAbs(ImageBuffer& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Buffer, cl::Buffer>(SelectProgram(Source), "init_abs")
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}


// Reductions
double StatisticsVector::Min(ImageBuffer& Source)
{
   Init(Source);

   Kernel(reduce_min, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double StatisticsVector::Max(ImageBuffer& Source)
{
   Init(Source);

   Kernel(reduce_max, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double StatisticsVector::MinAbs(ImageBuffer& Source)
{
   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double StatisticsVector::MaxAbs(ImageBuffer& Source)
{
   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result;
}

double StatisticsVector::Sum(ImageBuffer& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceSum(m_PartialResult);
}

uint StatisticsVector::CountNonZero(ImageBuffer& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_count_nz, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return (uint) ReduceSum(m_PartialResult);
}

double StatisticsVector::Mean(ImageBuffer& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

double StatisticsVector::MeanSqr(ImageBuffer& Source)
{
   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}


// Reductions that also find the coordinate
double StatisticsVector::Min(ImageBuffer& Source, int& outX, int& outY)
{
   PrepareCoords(Source);

   Kernel(min_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::Max(ImageBuffer& Source, int& outX, int& outY)
{
   PrepareCoords(Source);

   Kernel(max_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MinAbs(ImageBuffer& Source, int& outX, int& outY)
{
   PrepareCoords(Source);

   Kernel(min_abs_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MaxAbs(ImageBuffer& Source, int& outX, int& outY)
{
   PrepareCoords(Source);

   Kernel(max_abs_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}

}
