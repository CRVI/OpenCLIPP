////////////////////////////////////////////////////////////////////////////////
//! @file	: benchUnary.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a class for a unary primitive
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

#ifndef IPP_NAME
#define IPP_NAME BENCH_NAME
#endif

#ifndef IPP_PARAM_LAST
#define IPP_PARAM_LAST CONSTANT_LAST
#endif

#ifndef NPP_PARAM_LAST
#define NPP_PARAM_LAST CONSTANT_LAST
#endif

#ifndef CV_PARAM_LAST
#define CV_PARAM_LAST
#endif

#ifndef ADDITIONNAL_DECLARATIONS
#define ADDITIONNAL_DECLARATIONS
#endif

#define CLASS_NAME CONCATENATE(BENCH_NAME, Bench)

template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(BENCH_NAME, BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(BENCH_NAME, BenchU16);
#ifdef HAS_FLOAT
typedef CLASS_NAME<float>           CONCATENATE(BENCH_NAME, BenchF32);
#endif // HAS_FLOAT

template<typename DataType>
class CLASS_NAME : public BenchUnaryBase<DataType, USE_BUFFER>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

#ifndef CV_NAME
   bool HasCVTest() const { return false; }
#endif

   ADDITIONNAL_DECLARATIONS
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
#ifdef HAS_CL_BUFFER
   if (this->m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(this->m_CLBufferSrc, this->m_CLBufferDst CONSTANT_MIDDLE CONSTANT_LAST);
   else
#endif
      CONCATENATE(ocip, BENCH_NAME)(this->m_CLSrc, this->m_CLDst CONSTANT_MIDDLE CONSTANT_LAST);
}
#ifndef NO_UNSIGNED
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, IPP_NAME), _8u_C1R)(this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_MIDDLE, this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi IPP_PARAM_LAST);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, IPP_NAME), _16u_C1R)((Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_MIDDLE, (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi IPP_PARAM_LAST);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, IPP_NAME), _8u_C1R)((Npp8u*) this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_MIDDLE, (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi NPP_PARAM_LAST);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, IPP_NAME), _16u_C1R)((Npp16u*) this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_MIDDLE, (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi NPP_PARAM_LAST);
   )
}
#endif   // NO_UNSIGNED
//-----------------------------------------------------------------------------------------------------------------------------
#ifdef HAS_FLOAT
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, IPP_NAME), _32f_C1R)((Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_MIDDLE, (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi NPP_PARAM_LAST);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, IPP_NAME), _32f_C1R)((Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_MIDDLE, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi IPP_PARAM_LAST);
   )
}
#endif // HAS_FLOAT
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCV()
{
#ifdef CV_NAME
   CV_CODE( CV_NAME (m_CVSrc CONSTANT_MIDDLE CONSTANT_LAST , m_CVDst CV_PARAM_LAST ); )
#endif
}

#undef IPP_NAME
#undef IPP_PARAM_LAST
#undef NPP_PARAM_LAST
#undef CV_PARAM_LAST
#undef ADDITIONNAL_DECLARATIONS
#undef CLASS_NAME
#undef BENCH_NAME
#undef CV_NAME
