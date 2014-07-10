////////////////////////////////////////////////////////////////////////////////
//! @file	: benchCompareBase.hpp
//! @date   : Feb 2014
//!
//! @brief  : Creates classes for an unary image comapring 
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

#define CLASS_NAME CONCATENATE(CompareC, CONCATENATE(COMPARE_TYPE, Bench))
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(CLASS_NAME, U8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(CLASS_NAME, U16);
typedef CLASS_NAME<float>           CONCATENATE(CLASS_NAME, F32);


template<typename DataType>
class CLASS_NAME : public BenchUnaryBase<DataType>
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();

   bool HasNPPTest() const { return false; }
   bool HasCVTest() const { return false; }

   void Create(uint Width, uint Height);
};

template<typename DataType>
void CLASS_NAME<DataType>::Create(uint Width, uint Height)
{
	IBench1in1out::Create<DataType, unsigned char>(Width, Height);
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
			ippiCompareC_8u_C1R(this->m_ImgSrc.Data(), this->m_ImgSrc.Step, VALUE, this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
		)
}
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
			ippiCompareC_16u_C1R((Ipp16u*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, USHORT_VALUE, this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
		)
}
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
			ippiCompareC_32f_C1R((Ipp32f*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, FLOAT_VALUE, this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, GetIppCmpOp(COMPARE_TYPE));
		)
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
	float value = VALUE;
	if (is_same<DataType, unsigned short>::value)
		value = USHORT_VALUE;

	if (is_same<DataType, float>::value)
		value = FLOAT_VALUE;

   ocipCompareC(this->m_CLBufferSrc, this->m_CLBufferDst, value, COMPARE_TYPE);
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiCompareC_8u_C1R((Npp8u*)this->m_NPPSrc, this->m_NPPSrcStep, VALUE, (Npp8u*)this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
		)
}
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiCompareC_16u_C1R((Npp16u*)this->m_NPPSrc, this->m_NPPSrcStep, USHORT_VALUE, (Npp8u*)this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
		)
}
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
      nppiCompareC_32f_C1R((Npp32f*)this->m_NPPSrc, this->m_NPPSrcStep, FLOAT_VALUE, (Npp8u*)this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, GetNppCmpOp(COMPARE_TYPE));
		)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCV()
{

}

#undef CLASS_NAME
#undef COMPARE_TYPE
