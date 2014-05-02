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


#define KERNEL_RANGE(...)           GetRange(_FIRST(__VA_ARGS__))
#define LOCAL_RANGE                 GetLocalRange()
#define SELECT_NAME(name, src_img)  SelectName( #name , src_img)

#include "kernel_helpers.h"

#include "StatisticsHelpers.h"

#include <cmath>


using namespace std;

namespace OpenCLIPP
{


// StatisticsVector
void StatisticsVector::PrepareBuffer(const ImageBase& Image)
{
   size_t NbGroups = (size_t) GetNbGroups(Image);

   // We need space for 4 channels + another space for the number of pixels
   size_t BufferSize = NbGroups * (4 + 1);

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
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1, 1, 1)), Source, m_ResultBuffer);
}

void StatisticsVector::InitAbs(ImageBuffer& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Buffer, cl::Buffer>(SelectProgram(Source), "init_abs")
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1, 1, 1)), Source, m_ResultBuffer);
}


// Reductions
double StatisticsVector::Min(ImageBuffer& Source)
{
   Check1Channel(Source);

   Init(Source);

   Kernel(reduce_min, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::Max(ImageBuffer& Source)
{
   Check1Channel(Source);

   Init(Source);

   Kernel(reduce_max, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::MinAbs(ImageBuffer& Source)
{
   Check1Channel(Source);

   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::MaxAbs(ImageBuffer& Source)
{
   Check1Channel(Source);

   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::Sum(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceSum(m_PartialResult);
}

double StatisticsVector::SumSqr(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_sum_sqr, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceSum(m_PartialResult);
}

uint StatisticsVector::CountNonZero(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_count_nz, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return (uint) ReduceSum(m_PartialResult);
}

double StatisticsVector::Mean(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

double StatisticsVector::MeanSqr(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

double StatisticsVector::StdDev(ImageBuffer& Source)
{
   double mean;
   return StdDev(Source, mean);
}

double StatisticsVector::StdDev(ImageBuffer& Source, double& mean)
{
   mean = Mean(Source);

   Kernel(reduce_stddev, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height(), float(mean));

   m_PartialResultBuffer->Read(true);

   return sqrt(ReduceMean(m_PartialResult));
}


// Reductions that also find the coordinate
double StatisticsVector::Min(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(min_coord, In(Source), Out(), *m_PartialResultBuffer, *m_PartialCoordBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::Max(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(max_coord, In(Source), Out(), *m_PartialResultBuffer, *m_PartialCoordBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MinAbs(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(min_abs_coord, In(Source), Out(), *m_PartialResultBuffer, *m_PartialCoordBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MaxAbs(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(max_abs_coord, In(Source), Out(), *m_PartialResultBuffer, *m_PartialCoordBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}


// For images with multiple channels
void StatisticsVector::Min(ImageBuffer& Source, double outVal[4])
{
   Init(Source);

   Kernel(reduce_min, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (uint i = 0; i < Source.NbChannels(); i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::Max(ImageBuffer& Source, double outVal[4])
{
   Init(Source);

   Kernel(reduce_max, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (uint i = 0; i < Source.NbChannels(); i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::MinAbs(ImageBuffer& Source, double outVal[4])
{
   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (uint i = 0; i < Source.NbChannels(); i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::MaxAbs(ImageBuffer& Source, double outVal[4])
{
   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(), m_ResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (uint i = 0; i < Source.NbChannels(); i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::Sum(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceSum(m_PartialResult, Source.NbChannels(), outVal);
}

void StatisticsVector::SumSqr(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_sum_sqr, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceSum(m_PartialResult, Source.NbChannels(), outVal);
}

void StatisticsVector::Mean(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceMean(m_PartialResult, Source.NbChannels(),outVal);
}

void StatisticsVector::MeanSqr(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceMean(m_PartialResult, Source.NbChannels(), outVal);
}

void StatisticsVector::StdDev(ImageBuffer& Source, double outVal[4])
{
   double means[4] = {0};

   StdDev(Source, outVal, means);
}

void StatisticsVector::StdDev(ImageBuffer& Source, double outVal[4], double outMean[4])
{
   Mean(Source, outMean);

   cl_float4 fmeans = {{float(outMean[0]), float(outMean[1]), float(outMean[2]), float(outMean[3])}};
   
   Kernel(reduce_stddev, In(Source), Out(), *m_PartialResultBuffer, Source.Step(), Source.Width(), Source.Height(), fmeans);

   m_PartialResultBuffer->Read(true);

   ReduceMean(m_PartialResult, Source.NbChannels(), outVal);

   for (uint i = 0; i < Source.NbChannels(); i++)
      outVal[i] = sqrt(outVal[i]);
}

}
