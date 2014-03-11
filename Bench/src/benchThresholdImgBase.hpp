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
class CLASS_NAME : public BenchBinaryBase<DataType, THRESHOLD_USE_BUFFER>
{
public:
   void RunIPP();
   void RunCL();

   bool HasNPPTest()  const { return false; }
   bool HasCVTest()   const { return false; }
   bool HasCUDATest() const { return false; }
};

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunIPP()
{
	// IPP does not have this type of primitives
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
   if (UseBuffer())
	   CONCATENATE(ocip, BENCH_NAME)(m_CLSrc, m_CLSrcB, m_CLDst, COMPARE_TYPE);
   else
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _V)(m_CLBufferSrc, m_CLBufferSrcB, m_CLBufferDst, COMPARE_TYPE);
}

#undef CLASS_NAME
#undef COMPARE_TYPE
#undef BENCH_NAME
