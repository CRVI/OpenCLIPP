////////////////////////////////////////////////////////////////////////////////
//! @file	: benchImageProximityFFT.hpp
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
#define CLASS_NAME CONCATENATE(BENCH_NAME, Bench)
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(BENCH_NAME, BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(BENCH_NAME, BenchU16);
typedef CLASS_NAME<float>           CONCATENATE(BENCH_NAME, BenchF32);

#ifndef IPP_NAME
#define IPP_NAME BENCH_NAME
#endif

template<typename DataType>
class CLASS_NAME : public IBench1in1out
{
public:

   CLASS_NAME()
   :  IBench1in1out(true),
      m_Program(nullptr)
   { }

   void RunIPP();
   //void RunNPP();
   void RunCL();

   bool HasCUDATest() const { return false; }

   void Create(uint Width, uint Height);
   void Free();

   float CompareTolerance() const { return 0.01f; }
   bool CompareTolRelative() const { return true; }

protected:

   std::unique_ptr<CImageROI> m_ImgTemp;

   ocipBuffer m_TemplBuf;

   ocipProgram m_Program;

   IPP_CODE(
      IppiSize m_SrcSize;
      IppiSize m_TempSize;
      )
};

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, float>(Width, Height);

   // CL
   m_ImgTemp = std::unique_ptr<CImageROI>(new CImageROI(m_ImgSrc, 10, 10,
      min(100, m_ImgSrc.Width - 10), min(100, m_ImgSrc.Height - 10)));

   ocipCreateImageBuffer(&m_TemplBuf, m_ImgTemp->ToSImage(), m_ImgTemp->Data(), CL_MEM_READ_WRITE);

   ocipPrepareImageProximityFFT(&m_Program, m_CLBufferSrc, m_TemplBuf);

   // IPP
   IPP_CODE(
      m_SrcSize.width = m_ImgSrc.Width;
      m_SrcSize.height = m_ImgSrc.Height;
      m_TempSize.width = m_ImgTemp->Width;
      m_TempSize.height = m_ImgTemp->Height;
      )
  
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::Free()
{
   IBench1in1out::Free();

   ocipReleaseImageBuffer(m_TemplBuf);

   ocipReleaseProgram(m_Program);
   m_Program = nullptr;
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _8u32f_C1R)( this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                              this->m_SrcSize, this->m_ImgTemp->Data(), this->m_ImgTemp->Step, this->m_TempSize, 
                                                              (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
    IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _16u32f_C1R)( (Ipp16u*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                               this->m_SrcSize, (Ipp16u*)this->m_ImgTemp->Data(), this->m_ImgTemp->Step, this->m_TempSize, 
                                                               (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _32f_C1R)( (Ipp32f*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                            this->m_SrcSize, (Ipp32f*)this->m_ImgTemp->Data(), this->m_ImgTemp->Step, this->m_TempSize, 
                                                            (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
   CONCATENATE(ocip, BENCH_NAME)(m_Program, m_CLBufferSrc, m_TemplBuf, m_CLBufferDst);
}

#undef CLASS_NAME
#undef BENCH_NAME
#undef IPP_NAME
