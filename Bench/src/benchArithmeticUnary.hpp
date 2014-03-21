////////////////////////////////////////////////////////////////////////////////
//! @file	: benchArithmeticUnary.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a class for an arithmetic unary primitive
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
typedef CLASS_NAME<float>           CONCATENATE(BENCH_NAME, BenchF32);

template<typename DataType>
class CLASS_NAME : public BenchUnaryBase<DataType, USE_BUFFER>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

#ifdef NO_NEGATIVE
   void Create(uint Width, uint Height)
   {
      IBench1in1out::Create<DataType, DataType>(Width, Height, Width, Height, false);
   }
#endif

#ifndef CV_NAME
   bool HasCVTest() const { return false; }
#endif

#ifdef DIVIDE_TYPE_TOLERANCE
   // Divide and some other operations suffer from minor rounding differences
   // So we increase tolerance for these
   float CompareTolerance() const { return 1; }
   bool CompareTolRelative() const { return false; }
#endif   // DIVIDE_TYPE_TOLERANCE
};
//-----------------------------------------------------------------------------------------------------------------------------
#ifdef DIVIDE_TYPE_TOLERANCE
// Using relative tolerance for float images
template<>
float CLASS_NAME<float>::CompareTolerance() const
{
   return 0.00001f;
}
template<>
bool CLASS_NAME<float>::CompareTolRelative() const
{
   return true;
}
#endif   // DIVIDE_TYPE_TOLERANCE
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
   if (this->m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(this->m_CLBufferSrc, this->m_CLBufferDst CONSTANT_LAST);
   else
      CONCATENATE(ocip, BENCH_NAME)(this->m_CLSrc, this->m_CLDst CONSTANT_LAST);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _8u_C1RSfs)(this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_LAST, this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, 0);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _16u_C1RSfs)((Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_LAST, (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, 0);
   )
}
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), _32f_C1R)((Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step CONSTANT_LAST, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _8u_C1RSfs)((Npp8u*)this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_LAST, (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, 0);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _16u_C1RSfs)((Npp16u*)this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_LAST, (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, 0);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), _32f_C1R)((Npp32f*)this->m_NPPSrc, this->m_NPPSrcStep CONSTANT_LAST, (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);
   )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCV()
{
#ifdef CV_NAME
   CV_CODE( CV_NAME (m_CVSrc CONSTANT_LAST , m_CVDst); )
#endif
}

#undef CLASS_NAME
#undef BENCH_NAME
#undef CV_NAME
