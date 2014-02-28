////////////////////////////////////////////////////////////////////////////////
//! @file	: benchThresholdBase.hpp
//! @date   : Feb 2014
//!
//! @brief  : Base class for image thresholding
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

#define THRESH 100
#define VALUEGT 180
#define VALUELT 50

#define USHORT_THRESH 5000
#define USHORT_VALUEGT 6000
#define USHORT_VALUELT 2000

#define FLOAT_THRESH 0.2f
#define FLOAT_VALUEGT 0.7f
#define FLOAT_VALUELT -0.2f

#define CLASS_NAME CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), Bench)
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchU16);
typedef CLASS_NAME<float>           CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchF32);


template<typename DataType>
class CLASS_NAME : public BenchUnaryBase<DataType, false>
{
public:
   void RunIPP();
   void RunCUDA();
   void RunCL();
   void RunNPP();
   void RunCV();
};

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   if(THRESHOLD_TYPE == LT)
	{
		IPP_CODE(
			ippiThreshold_Val_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									  this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									  this->m_IPPRoi, THRESH, 
									  CONCATENATE(VALUE,THRESHOLD_TYPE),ippCmpLess);
		)
	}

	if(THRESHOLD_TYPE == GT)
	{
		IPP_CODE(
			ippiThreshold_Val_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									  this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									  this->m_IPPRoi, THRESH, 
									  CONCATENATE(VALUE,THRESHOLD_TYPE), ippCmpGreater);
		)
	}
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   if(THRESHOLD_TYPE == LT)
	{
		IPP_CODE(
			ippiThreshold_Val_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									   (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									   this->m_IPPRoi, USHORT_THRESH, 
									   CONCATENATE(USHORT_VALUE,THRESHOLD_TYPE),ippCmpLess);
		)
	}

	if(THRESHOLD_TYPE == GT)
	{
		IPP_CODE(
			ippiThreshold_Val_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									   (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									   this->m_IPPRoi, USHORT_THRESH, 
									   CONCATENATE(USHORT_VALUE,THRESHOLD_TYPE), ippCmpGreater);
		)
	}
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   if(THRESHOLD_TYPE == LT)
	{
		IPP_CODE(
			ippiThreshold_Val_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									   (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									   this->m_IPPRoi, FLOAT_THRESH, 
									   CONCATENATE(FLOAT_VALUE,THRESHOLD_TYPE),ippCmpLess);
		)
	}

	if(THRESHOLD_TYPE == GT)
	{
		IPP_CODE(
			ippiThreshold_Val_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
									   (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
									   this->m_IPPRoi, FLOAT_THRESH, 
									   CONCATENATE(FLOAT_VALUE,THRESHOLD_TYPE), ippCmpGreater);
		)
	}
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCUDA()
{

}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
	float thresh = THRESH;
	float valueGT = VALUEGT;
	float valueLT = VALUELT;

	if (is_same<DataType, unsigned short>::value)
	{
		thresh = USHORT_THRESH;
		valueGT = USHORT_VALUEGT;
		valueLT = USHORT_VALUELT;
	}
		

	if (is_same<DataType, float>::value)
	{
		thresh = FLOAT_THRESH;
		valueGT = FLOAT_VALUEGT;
		valueLT = FLOAT_VALUELT;
	}

	CONCATENATE(ocip, CONCATENATE(BENCH_NAME,THRESHOLD_TYPE))(m_CLSrc, m_CLDst, thresh, CONCATENATE(value,THRESHOLD_TYPE));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunNPP()
{

}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCV()
{

}

#undef CLASS_NAME
#undef THRESHOLD_TYPE
#undef BENCH_NAME
#undef THRESH 
#undef VALUEGT 
#undef VALUELT 