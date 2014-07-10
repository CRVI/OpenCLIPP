////////////////////////////////////////////////////////////////////////////////
//! @file	: benchBinary.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a class for a binary primitive
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

#define CLASS_NAME CONCATENATE(BENCH_NAME, Bench)

template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(BENCH_NAME, BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(BENCH_NAME, BenchU16);
#ifdef HAS_FLOAT
typedef CLASS_NAME<float>           CONCATENATE(BENCH_NAME, BenchF32);
#endif // HAS_FLOAT

template<typename DataType>
class CLASS_NAME : public BenchBinaryBase<DataType>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

#ifndef CV_NAME
   bool HasCVTest() const { return false; }
#endif
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
   CONCATENATE(ocip, BENCH_NAME) (this->m_CLBufferSrc, this->m_CLBufferSrcB, this->m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _8u_C1R)(
         this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
         this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
         this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _16u_C1R)(
         (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
         (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
         (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _8u_C1R)(
         (Npp8u*) this->m_NPPSrc, this->m_NPPSrcStep,
         (Npp8u*) this->m_NPPSrcB, this->m_NPPSrcBStep,
         (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _16u_C1R)(
         (Npp16u*) this->m_NPPSrc, this->m_NPPSrcStep,
         (Npp16u*) this->m_NPPSrcB, this->m_NPPSrcBStep,
         (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);
   )
}
#ifdef HAS_FLOAT
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _32f_C1R)(
         (Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep,
         (Npp32f*) this->m_NPPSrcB, this->m_NPPSrcBStep,
         (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _32f_C1R)(
         (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
         (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
         (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);
   )
}
#endif // HAS_FLOAT
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCV()
{
   CV_CODE( CV_NAME (m_CVSrcB, m_CVSrc, m_CVDst); )
}

#undef CLASS_NAME
#undef BENCH_NAME
#undef CV_NAME
