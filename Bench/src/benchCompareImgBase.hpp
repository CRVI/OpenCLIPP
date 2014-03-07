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

//#define BENCH_NAME ThresholdGT  _Img
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
   void RunCUDA();
   void RunCL();
   void RunNPP();
   void RunCV();
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
	if(COMPARE_TYPE == LT)
	{
		IPP_CODE(
			ippiCompare_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								this->m_IPPRoi, ippCmpLess);
		)
	}
	if(COMPARE_TYPE == LQ)
	{
		IPP_CODE(
			ippiCompare_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								this->m_IPPRoi, ippCmpLessEq);
		)
	}
	if(COMPARE_TYPE == EQ)
	{
		IPP_CODE(
			ippiCompare_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								this->m_IPPRoi, ippCmpEq);
		)
	}
	if(COMPARE_TYPE == GT)
	{
		IPP_CODE(
			ippiCompare_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								this->m_IPPRoi, ippCmpGreater);
		)
	}
	if(COMPARE_TYPE == GQ)
	{
		IPP_CODE(
			ippiCompare_8u_C1R( this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								this->m_IPPRoi, ippCmpGreaterEq);
		)
	}
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
	if(COMPARE_TYPE == LT)
	{
		IPP_CODE(
			ippiCompare_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpLess);
		)
	}
	if(COMPARE_TYPE == LQ)
	{
		IPP_CODE(
			ippiCompare_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpLessEq);
		)
	}
	if(COMPARE_TYPE == EQ)
	{
		IPP_CODE(
			ippiCompare_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpEq);
		)
	}
	if(COMPARE_TYPE == GT)
	{
		IPP_CODE(
			ippiCompare_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpGreater);
		)
	}
	if(COMPARE_TYPE == GQ)
	{
		IPP_CODE(
			ippiCompare_16u_C1R( (Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp16u*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpGreaterEq);
		)
	}
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
	if(COMPARE_TYPE == LT)
	{
		IPP_CODE(
			ippiCompare_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpLess);
		)
	}
	if(COMPARE_TYPE == LQ)
	{
		IPP_CODE(
			ippiCompare_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpLessEq);
		)
	}
	if(COMPARE_TYPE == EQ)
	{
		IPP_CODE(
			ippiCompare_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpEq);
		)
	}
	if(COMPARE_TYPE == GT)
	{
		IPP_CODE(
			ippiCompare_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpGreater);
		)
	}
	if(COMPARE_TYPE == GQ)
	{
		IPP_CODE(
			ippiCompare_32f_C1R( (Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step,
								 (Ipp32f*) this->m_ImgSrcB.Data(), this->m_ImgSrcB.Step,
								 this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, 
								 this->m_IPPRoi, ippCmpGreaterEq);
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
   if (m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(m_CLBufferSrc, m_CLBufferSrcB, m_CLBufferDst, COMPARE_TYPE);
   else
      CONCATENATE(ocip, BENCH_NAME)(m_CLSrc, m_CLSrcB, m_CLDst, COMPARE_TYPE);
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
#undef COMPARE_TYPE
#undef BENCH_NAME