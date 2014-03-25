////////////////////////////////////////////////////////////////////////////////
//! @file	: benchLut.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for LUT
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

template<typename DataType> class LutBench;

typedef LutBench<unsigned char>  LutBenchU8;


template<typename DataType>
class LutBench : public IBench1in1out
{
public:
   LutBench()
   :  IBench1in1out(USE_BUFFER)
   { }

   void Create(uint Width, uint Height);
   void Free();
   void RunIPP();
   void RunCL();
   void RunNPP();

   bool HasCVTest() const { return false; }

   const static int Length = 5;

private:
   vector<uint> m_Levels;
   vector<uint> m_Values;

   NPP_CODE(
      Npp32s * m_NPPLevels;
      Npp32s * m_NPPValues;
      )
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LutBench<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, DataType>(Width, Height);

   m_Levels.assign(Length, 0);
   m_Values.assign(Length, 0);

   for (uint& e : m_Levels)
      e = rand() % 256;

   for (uint& e : m_Values)
      e = rand() % 256;

   sort(m_Levels.begin(), m_Levels.end());
   sort(m_Values.begin(), m_Values.end());


   NPP_CODE(
      cudaMalloc((void**) &m_NPPLevels, Length * sizeof(Npp32s));
      cudaMalloc((void**) &m_NPPValues, Length * sizeof(Npp32s));

      cudaMemcpy(m_NPPLevels, m_Levels.data(), Length * sizeof(Npp32s), cudaMemcpyHostToDevice);
      cudaMemcpy(m_NPPValues, m_Values.data(), Length * sizeof(Npp32s), cudaMemcpyHostToDevice);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LutBench<DataType>::Free()
{
   IBench1in1out::Free();

   NPP_CODE(
      cudaFree(m_NPPLevels);
      cudaFree(m_NPPValues);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LutBench<DataType>::RunIPP()
{
   IPP_CODE(
      ippiLUT_8u_C1R(m_ImgSrc.Data(), m_ImgSrc.Step, m_ImgDstIPP.Data(), m_ImgDstIPP.Step,
         m_IPPRoi, (int*) m_Values.data(), (int*) m_Levels.data(), Length);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void LutBench<DataType>::RunCL()
{
   if (CLUsesBuffer())
      ocipLut_V(m_CLBufferSrc, m_CLBufferDst, m_Levels.data(), m_Values.data(), Length);
   else
      ocipLut(m_CLSrc, m_CLDst, m_Levels.data(), m_Values.data(), Length);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void LutBench<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiLUT_8u_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi, m_NPPValues, m_NPPLevels, Length);
      )
}
