////////////////////////////////////////////////////////////////////////////////
//! @file	: benchLinearLut.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for linear LUT
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

#include <algorithm>
#include <vector>

using namespace std;

template<typename DataType> class LinearLutBench;

typedef LinearLutBench<float>  LinearLutBenchF32;


template<typename DataType>
class LinearLutBench : public IBench1in1out
{
public:
   LinearLutBench()
   :  IBench1in1out(USE_BUFFER)
   { }

   void Create(uint Width, uint Height);
   void Free();
   void RunIPP();
   void RunCL();
   void RunNPP();

   bool HasCVTest() const { return false; }

   typedef DataType dataType;

   const static int Length = 5;

private:
   vector<float> m_Levels;
   vector<float> m_Values;

   NPP_CODE(
      Npp32f * m_NPPLevels;
      Npp32f * m_NPPValues;
      )
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LinearLutBench<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, DataType>(Width, Height);

   m_Levels.assign(Length, 0);
   m_Values.assign(Length, 0);

   for (float& e : m_Levels)
      e = static_cast<float>(rand() % 256);

   for (float& e : m_Values)
      e = static_cast<float>(rand() % 256);

   sort(m_Levels.begin(), m_Levels.end());
   sort(m_Values.begin(), m_Values.end());


   NPP_CODE(
      cudaMalloc((void**) &m_NPPLevels, Length * sizeof(Npp32f));
      cudaMalloc((void**) &m_NPPValues, Length * sizeof(Npp32f));

      cudaMemcpy(m_NPPLevels, m_Levels.data(), Length * sizeof(Npp32f), cudaMemcpyHostToDevice);
      cudaMemcpy(m_NPPValues, m_Values.data(), Length * sizeof(Npp32f), cudaMemcpyHostToDevice);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LinearLutBench<DataType>::Free()
{
   IBench1in1out::Free();

   NPP_CODE(
      cudaFree(m_NPPLevels);
      cudaFree(m_NPPValues);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LinearLutBench<DataType>::RunIPP()
{
   IPP_CODE(
      ippiLUT_Linear_32f_C1R((Ipp32f*) m_ImgSrc.Data(), m_ImgSrc.Step, (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step,
         m_IPPRoi, m_Values.data(), m_Levels.data(), Length);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LinearLutBench<DataType>::RunCL()
{
   if (CLUsesBuffer())
      ocipLutLinear_V(m_CLBufferSrc, m_CLBufferDst, m_Levels.data(), m_Values.data(), Length);
   else
      ocipLutLinear(m_CLSrc, m_CLDst, m_Levels.data(), m_Values.data(), Length);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void LinearLutBench<float>::RunNPP()
{
   NPP_CODE(
      nppiLUT_Linear_32f_C1R((Npp32f*) m_NPPSrc, m_NPPSrcStep, (Npp32f*) m_NPPDst, m_NPPDstStep,
         m_NPPRoi, m_NPPValues, m_NPPLevels, Length);
      )
}
