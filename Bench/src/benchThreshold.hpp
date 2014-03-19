////////////////////////////////////////////////////////////////////////////////
//! @file	: benchThreshold.hpp
//! @date   : Feb 2014
//!
//! @brief  : Creates classes for image thresholding
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

#define THRESHOLD_USE_BUFFER USE_BUFFER

IPP_CODE(IppCmpOp GetIppCmpOp(ECompareOperation Op) { return IppCmpOp(Op); } )   // Same numerical value
NPP_CODE(NppCmpOp GetNppCmpOp(ECompareOperation Op) { return NppCmpOp(Op); } )   // Same numerical value

#define THRESH 100
#define VALUEGT 180
#define VALUELT 50

#define USHORT_THRESH 5000
#define USHORT_VALUEGT 6000
#define USHORT_VALUELT 2000

#define FLOAT_THRESH 0.2f
#define FLOAT_VALUEGT 0.7f
#define FLOAT_VALUELT -0.2f


template<typename T>
float GetThreshold()
{
   if (is_same<T, unsigned short>::value)
      return USHORT_THRESH;

   if (is_same<T, float>::value)
      return FLOAT_THRESH;

   return THRESH;
}

template<typename T>
float GetThresholdValue(ECompareOperation Op)
{
   if (Op < EQ)
   {
      if (is_same<T, unsigned short>::value)
         return USHORT_VALUELT;

      if (is_same<T, float>::value)
         return FLOAT_VALUELT;

      return VALUELT;
   }

   if (is_same<T, unsigned short>::value)
      return USHORT_VALUEGT;

   if (is_same<T, float>::value)
      return FLOAT_VALUEGT;

   return VALUEGT;
}

#define THRESHOLD_TYPE LT
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"

#define THRESHOLD_TYPE LQ
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"

#define THRESHOLD_TYPE EQ
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"

#define THRESHOLD_TYPE GQ
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"

#define THRESHOLD_TYPE GT
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"


#define COMPARE_TYPE LT
#define BENCH_NAME Threshold_Img
#include "benchThresholdImgBase.hpp"

#define COMPARE_TYPE LQ
#define BENCH_NAME Threshold_Img
#include "benchThresholdImgBase.hpp"

#define COMPARE_TYPE EQ
#define BENCH_NAME Threshold_Img
#include "benchThresholdImgBase.hpp"

#define COMPARE_TYPE GQ
#define BENCH_NAME Threshold_Img
#include "benchThresholdImgBase.hpp"

#define COMPARE_TYPE GT
#define BENCH_NAME Threshold_Img
#include "benchThresholdImgBase.hpp"
