////////////////////////////////////////////////////////////////////////////////
//! @file	: benchImageProximity.hpp
//! @date   : Mar 2014
//!
//! @brief  : Creates a benchmark class for Image Proximity operations
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

#define CLASS_NAME CONCATENATE(BENCH_NAME, Bench)
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(BENCH_NAME, BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(BENCH_NAME, BenchU16);
typedef CLASS_NAME<float>           CONCATENATE(BENCH_NAME, BenchF32);

template<typename DataType>
class CLASS_NAME : public IBench1in1out
{
public:

   CLASS_NAME(): IBench1in1out(PROXIMITY_USE_BUFFER)
   { }

   void RunIPP();
   //void RunNPP();
   void RunCL();

   bool HasCVTest()   const { return false; }
   bool HasNPPTest()  const { return false; }

   void Create(uint Width, uint Height);
   void Free();
 
protected:

   ocipImage m_CLTmp;
   std::unique_ptr<CImageROI> m_ImgTemp;

   IPP_CODE(
      IppiSize m_SrcSize;
      IppiSize m_TempSize;
      )
};

template<typename DataType>
inline void CLASS_NAME<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, float>(Width, Height);

   // CL
   m_ImgTemp = std::unique_ptr<CImageROI>(new CImageROI(m_ImgSrc, 10, 10,
      min(16, int(m_ImgSrc.Width) - 10), min(16, int(m_ImgSrc.Height) - 10)));

   ocipCreateImage(&m_CLTmp, m_ImgTemp->ToSImage(), m_ImgTemp->Data(), CL_MEM_READ_WRITE);

   // IPP
   IPP_CODE(
      m_SrcSize.width = m_ImgSrc.Width;
      m_SrcSize.height = m_ImgSrc.Height;
      m_TempSize.width = m_ImgTemp->Width;
      m_TempSize.height = m_ImgTemp->Height;
      )
}

template<typename DataType>
inline void CLASS_NAME<DataType>::Free()
{
   IBench1in1out::Free();

   ocipReleaseImage(m_CLTmp);
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
      ippiSqrDistanceSame_Norm_8u32f_C1R( m_ImgSrc.Data(), m_ImgSrc.Step, 
                                          m_SrcSize, m_ImgTemp->Data(), m_ImgTemp->Step, m_TempSize, 
                                          (Ipp32f*)m_ImgDstIPP.Data(), m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
         ippiSqrDistanceSame_Norm_16u32f_C1R((Ipp16u*)m_ImgSrc.Data(), m_ImgSrc.Step, m_SrcSize, 
                                             (Ipp16u*)m_ImgTemp->Data(), m_ImgTemp->Step, m_TempSize, 
                                             (Ipp32f*)m_ImgDstIPP.Data(), m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
         ippiSqrDistanceSame_Norm_32f_C1R((Ipp32f*)m_ImgSrc.Data(), m_ImgSrc.Step, m_SrcSize, 
                                          (Ipp32f*)m_ImgTemp->Data(), m_ImgTemp->Step, m_TempSize, 
                                          (Ipp32f*)m_ImgDstIPP.Data(), m_ImgDstIPP.Step);
      )
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{  
   CONCATENATE(ocip, BENCH_NAME)(m_CLSrc, m_CLTmp, m_CLDst);
}

#undef CLASS_NAME
