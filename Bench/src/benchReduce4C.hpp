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

template<typename DataType> class CONCATENATE(BENCH_NAME, 4CBench);

typedef CONCATENATE(BENCH_NAME, 4CBench)<unsigned char>    CONCATENATE(BENCH_NAME, 4CBenchU8);
typedef CONCATENATE(BENCH_NAME, 4CBench)<unsigned short>   CONCATENATE(BENCH_NAME, 4CBenchU16);
typedef CONCATENATE(BENCH_NAME, 4CBench)<float>            CONCATENATE(BENCH_NAME, 4CBenchF32);

template<typename DataType>
class CONCATENATE(BENCH_NAME, 4CBench) : public BenchReduceBase<DataType, REDUCE_DST_TYPE>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();

   void Create(uint Width, uint Height);

   bool HasCVTest() const { return false; }

   float CompareTolerance() const { return REDUCE_CMP_TOLERANCE; }
};
template<typename DataType>
void CONCATENATE(BENCH_NAME, 4CBench)<DataType>::Create(uint Width, uint Height)
{
   BenchReduceBase<DataType, REDUCE_DST_TYPE>::Create(Width, Height, 4);
}
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<unsigned char>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _8u_C4R)(
         this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, this->m_DstIPP IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _16u_C4R)(
         (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, this->m_DstIPP IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _32f_C4R)(
         (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, this->m_DstIPP IPP_REDUCE_HINT IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CONCATENATE(BENCH_NAME, 4CBench)<DataType>::RunCL()
{
   if (this->m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(this->m_Program, this->m_CLBufferSrc, this->m_DstCL CL_ADDITIONAL_PARAMS);
   else
      CONCATENATE(ocip, BENCH_NAME)(this->m_Program, this->m_CLSrc, this->m_DstCL CL_ADDITIONAL_PARAMS);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _8u_C4R)(
         (Npp8u*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _16u_C4R)(
         (Npp16u*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, 4CBench)<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _32f_C4R)(
         (Npp32f*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}

#undef BENCH_NAME
