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

#define CLASS_NAME CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), Bench)
template<typename DataType> class CLASS_NAME;

typedef CLASS_NAME<unsigned char>   CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchU8);
typedef CLASS_NAME<unsigned short>  CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchU16);
typedef CLASS_NAME<float>           CONCATENATE(CONCATENATE(BENCH_NAME,THRESHOLD_TYPE), BenchF32);


template<typename DataType>
class CLASS_NAME : public BenchUnaryBase<DataType, THRESHOLD_USE_BUFFER>
{
public:
   void RunIPP();
   void RunNPP();
   void RunCL();

   bool HasCVTest()   const { return false; }
   bool HasCUDATest() const { return false; }
};

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
			ippiThreshold_Val_8u_C1R( m_ImgSrc.Data(), m_ImgSrc.Step,
									  m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
									  m_IPPRoi, THRESH, 
									  CONCATENATE(VALUE,THRESHOLD_TYPE), GetIppCmpOp(THRESHOLD_TYPE));
		)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
   IPP_CODE(
			ippiThreshold_Val_16u_C1R( (Ipp16u*) m_ImgSrc.Data(), m_ImgSrc.Step,
									   (Ipp16u*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
									   m_IPPRoi, USHORT_THRESH, 
									   CONCATENATE(USHORT_VALUE,THRESHOLD_TYPE), GetIppCmpOp(THRESHOLD_TYPE));
		)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
			ippiThreshold_Val_32f_C1R( (Ipp32f*) m_ImgSrc.Data(), m_ImgSrc.Step,
									   (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, 
									   m_IPPRoi, FLOAT_THRESH, 
									   CONCATENATE(FLOAT_VALUE,THRESHOLD_TYPE), GetIppCmpOp(THRESHOLD_TYPE));
		)
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

   if (m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, CONCATENATE(BENCH_NAME,THRESHOLD_TYPE)), _V)(m_CLBufferSrc, m_CLBufferDst, thresh, CONCATENATE(value,THRESHOLD_TYPE));
   else
      CONCATENATE(ocip, CONCATENATE(BENCH_NAME,THRESHOLD_TYPE))(m_CLSrc, m_CLDst, thresh, CONCATENATE(value,THRESHOLD_TYPE));
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
         nppiThreshold_Val_8u_C1R( (Npp8u*) m_NPPSrc, m_NPPSrcStep,
                             (Npp8u*) m_NPPDst, m_NPPDstStep,
									  m_NPPRoi, THRESH, 
									  CONCATENATE(VALUE,THRESHOLD_TYPE), GetNppCmpOp(THRESHOLD_TYPE));
		)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
   NPP_CODE(
			nppiThreshold_Val_16u_C1R( (Ipp16u*) m_NPPSrc, m_NPPSrcStep,
									   (Ipp16u*) m_NPPDst, m_NPPDstStep,
									   m_NPPRoi, USHORT_THRESH, 
									   CONCATENATE(USHORT_VALUE,THRESHOLD_TYPE), GetNppCmpOp(THRESHOLD_TYPE));
		)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
			nppiThreshold_Val_32f_C1R( (Ipp32f*) m_NPPSrc, m_NPPSrcStep,
									   (Ipp32f*) m_NPPDst, m_NPPDstStep,
									   m_NPPRoi, FLOAT_THRESH, 
									   CONCATENATE(FLOAT_VALUE,THRESHOLD_TYPE), GetNppCmpOp(THRESHOLD_TYPE));
		)
}


#undef CLASS_NAME
#undef THRESHOLD_TYPE
#undef BENCH_NAME
