////////////////////////////////////////////////////////////////////////////////
//! @file	: benchReduce.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a benchmark class for reduction operations
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

template<typename DataType> class CONCATENATE(BENCH_NAME, Bench);

typedef CONCATENATE(BENCH_NAME, Bench)<unsigned char>    CONCATENATE(BENCH_NAME, BenchU8);
typedef CONCATENATE(BENCH_NAME, Bench)<unsigned short>   CONCATENATE(BENCH_NAME, BenchU16);
typedef CONCATENATE(BENCH_NAME, Bench)<float>            CONCATENATE(BENCH_NAME, BenchF32);

template<typename DataType>
class CONCATENATE(BENCH_NAME, Bench) : public BenchReduceBase<DataType, REDUCE_DST_TYPE>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

#ifndef CV_OPERATION
   bool HasCVTest() const { return false; }
#endif

   float CompareTolerance() const { return REDUCE_CMP_TOLERANCE; }
};
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned char>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _8u_C1R)(
         this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, MIDDLE_PARAM &this->m_DstIPP[0] IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _16u_C1R)(
         (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, MIDDLE_PARAM &this->m_DstIPP[0] IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _32f_C1R)(
         (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, MIDDLE_PARAM &this->m_DstIPP[0] IPP_REDUCE_HINT IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CONCATENATE(BENCH_NAME, Bench)<DataType>::RunCL()
{
   CONCATENATE(ocip, BENCH_NAME) (this->m_Program, this->m_CLSrc, MIDDLE_PARAM this->m_DstCL CL_ADDITIONAL_PARAMS);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _8u_C1R)(
         (Npp8u*) this->m_NPPSrc, this->m_NPPSrcStep, this->m_NPPRoi, this->m_NPPWorkBuffer, NPP_MIDDLE_PARAM this->m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _16u_C1R)(
         (Npp16u*) this->m_NPPSrc, this->m_NPPSrcStep, this->m_NPPRoi, this->m_NPPWorkBuffer, NPP_MIDDLE_PARAM this->m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _32f_C1R)(
         (Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep, this->m_NPPRoi, this->m_NPPWorkBuffer, NPP_MIDDLE_PARAM this->m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CONCATENATE(BENCH_NAME, Bench)<DataType>::RunCV()
{
#ifdef CV_OPERATION
   CV_CODE(CV_OPERATION(m_CVSrc, m_DstCV);)
#endif
}

#undef BENCH_NAME
#undef CV_OPERATION
