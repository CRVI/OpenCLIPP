////////////////////////////////////////////////////////////////////////////////
//! @file	: benchThresholdCompareImgBase.hpp
//! @date   : Feb 2014
//!
//! @brief  : Creates classes for an binary image comparing and thresholding 
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


#define CLASS_NAME CONCATENATE(BENCH_NAME, CONCATENATE(COMPARE_TYPE, Bench))
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(CONCATENATE(BENCH_NAME, COMPARE_TYPE), BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(CONCATENATE(BENCH_NAME, COMPARE_TYPE), BenchU16);
typedef CLASS_NAME<float>           CONCATENATE(CONCATENATE(BENCH_NAME, COMPARE_TYPE), BenchF32);


template<typename DataType>
class CLASS_NAME : public BenchBinaryBase<DataType, COMPARE_USE_BUFFER>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();

   bool HasCVTest()   const { return false; }
   bool HasCUDATest() const { return false; }

   void Create(uint Width, uint Height);
};

template<typename DataType>
void CLASS_NAME<DataType>::Create(uint Width, uint Height)
{
   IBench2in1out::Create<DataType, unsigned char>(Width, Height);
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
         ippiCompare_8u_C1R( m_ImgSrc.Data(), m_ImgSrc.Step,
                        m_ImgSrcB.Data(), m_ImgSrcB.Step,
                        m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
                        m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
         ippiCompare_16u_C1R( (Ipp16u*) m_ImgSrc.Data(), m_ImgSrc.Step,
                         (Ipp16u*) m_ImgSrcB.Data(), m_ImgSrcB.Step,
                         m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
                         m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
         ippiCompare_32f_C1R( (Ipp32f*) m_ImgSrc.Data(), m_ImgSrc.Step,
                         (Ipp32f*) m_ImgSrcB.Data(), m_ImgSrcB.Step,
                         m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
                         m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
      )
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
   if (m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(m_CLBufferSrc, m_CLBufferSrcB, m_CLBufferDst, COMPARE_TYPE);
   else
      CONCATENATE(ocip, BENCH_NAME)(m_CLSrc, m_CLSrcB, m_CLDst, COMPARE_TYPE);
}


//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
         nppiCompare_8u_C1R( (Npp8u*) m_NPPSrc, m_NPPSrcStep,
                        (Npp8u*) m_NPPSrcB, m_NPPSrcBStep,
                        (Npp8u*) m_NPPDst, m_NPPDstStep,
                        m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
         nppiCompare_16u_C1R( (Ipp16u*) m_NPPSrc, m_NPPSrcStep,
                         (Ipp16u*) m_NPPSrcB, m_NPPSrcBStep,
                         (Npp8u*) m_NPPDst, m_NPPDstStep,
                         m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
         nppiCompare_32f_C1R( (Ipp32f*) m_NPPSrc, m_NPPSrcStep,
                         (Ipp32f*) m_NPPSrcB, m_NPPSrcBStep,
                         (Npp8u*) m_NPPDst, m_NPPDstStep,
                         m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
      )
}


#undef CLASS_NAME
#undef COMPARE_TYPE
#undef BENCH_NAME
