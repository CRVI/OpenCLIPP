////////////////////////////////////////////////////////////////////////////////
//! @file	: benchCompare.hpp
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

//enum ECompareOperation { LT, LQ, EQ, GQ, GT, };
#define COMPARE_USE_BUFFER true

#define VALUE 100
#define USHORT_VALUE 5000
#define FLOAT_VALUE 0.5

#define COMPARE_TYPE LT
#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)
#include "benchCompareBase.hpp"

#define COMPARE_TYPE LQ
#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)
#include "benchCompareBase.hpp"

#define COMPARE_TYPE EQ
#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)
#include "benchCompareBase.hpp"

#define COMPARE_TYPE GQ
#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)
#include "benchCompareBase.hpp"

#define COMPARE_TYPE GT
#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)
#include "benchCompareBase.hpp"
//
//
//#define BENCH_NAME CONCATENATE(Compare, COMPARE_TYPE)

//#define BENCH_NAME CompareLT
//#include "benchCompareBase.hpp"
//
//#define BENCH_NAME CompareLQ
//#include "benchCompareBase.hpp"
//
//#define BENCH_NAME CompareEQ
//#include "benchCompareBase.hpp"
//
//#define BENCH_NAME CompareGQ
//#include "benchCompareBase.hpp"
//
//#define BENCH_NAME CompareGT
//#include "benchCompareBase.hpp"

//template<typename DataType>
//class CompareBenchBase : public BenchUnaryBase<DataType, false>
//{
//public:
//   void Create(uint Width, uint Height);
//};

#define COMPARE_TYPE LT
#define BENCH_NAME Compare_Img
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE LQ
#define BENCH_NAME Compare_Img
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE EQ
#define BENCH_NAME Compare_Img
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE GQ
#define BENCH_NAME Compare_Img
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE GT
#define BENCH_NAME Compare_Img
#include "benchCompareImgBase.hpp"