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
#define SELECT_NAME(name, src_img) SelectSVName( #name , src_img)

#include "kernel_helpers.h"

#include "StatisticsHelpers.h"


using namespace std;

namespace OpenCLIPP
{

string SelectSVName(const char * name, const ImageBase& Image);


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

   cl::make_kernel<cl::Buffer, cl::Buffer>(SelectProgram(Source), SelectSVName("init", Source))
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}

void StatisticsVector::InitAbs(ImageBuffer& Source)
{
   Source.SendIfNeeded();

   cl::make_kernel<cl::Buffer, cl::Buffer>(SelectProgram(Source), SelectSVName("init_abs", Source))
      (cl::EnqueueArgs(*m_CL, cl::NDRange(1)), Source, m_ResultBuffer);
}


// Reductions
double StatisticsVector::Min(ImageBuffer& Source)
{
   Check1Channel(Source);

   Init(Source);

   Kernel(reduce_min, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::Max(ImageBuffer& Source)
{
   Check1Channel(Source);

   Init(Source);

   Kernel(reduce_max, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::MinAbs(ImageBuffer& Source)
{
   Check1Channel(Source);

   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::MaxAbs(ImageBuffer& Source)
{
   Check1Channel(Source);

   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   return m_Result[0];
}

double StatisticsVector::Sum(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceSum(m_PartialResult);
}

uint StatisticsVector::CountNonZero(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_count_nz, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return (uint) ReduceSum(m_PartialResult);
}

double StatisticsVector::Mean(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}

double StatisticsVector::MeanSqr(ImageBuffer& Source)
{
   Check1Channel(Source);

   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   return ReduceMean(m_PartialResult);
}


// Reductions that also find the coordinate
double StatisticsVector::Min(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(min_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::Max(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(max_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MinAbs(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(min_abs_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMin(m_PartialResult, m_PartialCoord, outX, outY);
}

double StatisticsVector::MaxAbs(ImageBuffer& Source, int& outX, int& outY)
{
   Check1Channel(Source);

   PrepareCoords(Source);

   Kernel(max_abs_coord, In(Source), Out(*m_PartialResultBuffer, *m_PartialCoordBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read();
   m_PartialCoordBuffer->Read(true);

   return ReduceMax(m_PartialResult, m_PartialCoord, outX, outY);
}


// For images with multiple channels
void StatisticsVector::Min(ImageBuffer& Source, double outVal[4])
{
   Init(Source);

   Kernel(reduce_min, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (int i = 0; i < 4; i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::Max(ImageBuffer& Source, double outVal[4])
{
   Init(Source);

   Kernel(reduce_max, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (int i = 0; i < 4; i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::MinAbs(ImageBuffer& Source, double outVal[4])
{
   InitAbs(Source);

   Kernel(reduce_minabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (int i = 0; i < 4; i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::MaxAbs(ImageBuffer& Source, double outVal[4])
{
   InitAbs(Source);

   Kernel(reduce_maxabs, In(Source), Out(m_ResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_ResultBuffer.Read(true);

   for (int i = 0; i < 4; i++)
      outVal[i] = m_Result[i];
}

void StatisticsVector::Sum(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_sum, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceSum_4C(m_PartialResult, outVal);
}

void StatisticsVector::Mean(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_mean, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceMean_4C(m_PartialResult, outVal);
}

void StatisticsVector::MeanSqr(ImageBuffer& Source, double outVal[4])
{
   PrepareBuffer(Source);

   Kernel(reduce_mean_sqr, In(Source), Out(*m_PartialResultBuffer), Source.Step(), Source.Width(), Source.Height());

   m_PartialResultBuffer->Read(true);

   ReduceMean_4C(m_PartialResult, outVal);
}


// Select the proper kernel name
string SelectSVName(const char * name, const ImageBase& Image)
{
   string Name = name;

   switch (Image.NbChannels())
   {
   case 1:
      break;
   case 2:
      Name += "_2C";
      break;
   case 3:
      Name += "_3C";
      break;
   case 4:
      Name += "_4C";
      break;
   default:
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "images must have between 1 and 4 channels");
      break;
   }
   
   bool InitKernel = (string(name).find("init") != string::npos);

   if (!InitKernel && IsFlushImage(Image))
      Name += "_flush";       // Use faster version

   return Name;
}

}
