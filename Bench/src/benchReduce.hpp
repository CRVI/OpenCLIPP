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
   void RunCUDA();
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
         this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, &this->m_DstIPP IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _16u_C1R)(
         (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, &this->m_DstIPP IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _32f_C1R)(
         (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, this->m_IPPRoi, &this->m_DstIPP IPP_REDUCE_HINT IPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CONCATENATE(BENCH_NAME, Bench)<DataType>::RunCL()
{
   if (this->m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(this->m_Program, this->m_CLBufferSrc, &this->m_DstCL CL_ADDITIONAL_PARAMS);
   else
      CONCATENATE(ocip, BENCH_NAME)(this->m_Program, this->m_CLSrc, &this->m_DstCL CL_ADDITIONAL_PARAMS);
}
//-----------------------------------------------------------------------------------------------------------------------------
#ifdef CUDA_REDUCE_SAME_TYPE
template<typename DataType>
void CONCATENATE(BENCH_NAME, Bench)<DataType>::RunCUDA()
{
   CUDA_CODE(
      DataType* CUDADst = reinterpret_cast<DataType*>(this->m_CUDADst.DeviceRef());
      CUDAPP(BENCH_NAME<DataType>)(Src(), m_CUDASrcStep, CUDADst, m_ImgSrc.Width, m_ImgSrc.Height, m_CUDAWorkBuffer);
      )
}
#else // CUDA_REDUCE_SAME_TYPE
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CONCATENATE(BENCH_NAME, Bench)<DataType>::RunCUDA()
{
   CUDA_CODE(
      uint* CUDADst = reinterpret_cast<uint*>(m_CUDADst.DeviceRef());
      CUDAPP(BENCH_NAME<DataType>)(Src(), m_CUDASrcStep, CUDADst, m_ImgSrc.Width, m_ImgSrc.Height, m_CUDAWorkBuffer);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<float>::RunCUDA()
{
   CUDA_CODE(
      float* CUDADst = reinterpret_cast<float*>(m_CUDADst.DeviceRef());
      CUDAPP(BENCH_NAME<float>)(Src(), m_CUDASrcStep, CUDADst, m_ImgSrc.Width, m_ImgSrc.Height, m_CUDAWorkBuffer);
      )
}
#endif   // CUDA_REDUCE_SAME_TYPE
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _8u_C1R)(
         (Npp8u*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _16u_C1R)(
         (Npp16u*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CONCATENATE(BENCH_NAME, Bench)<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _32f_C1R)(
         (Npp32f*) m_NPPSrc, m_NPPSrcStep, m_NPPRoi, m_NPPWorkBuffer, m_NPPDst NPP_ADDITIONAL_PARAMS);
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
