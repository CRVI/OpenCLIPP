////////////////////////////////////////////////////////////////////////////////
//! @file	: benchStatistics.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates classes for all supported statistical reductions
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

#include "benchReduceBase.hpp"

#define REDUCE_DST_TYPE DataType
#define IPP_REDUCE_HINT
#define REDUCE_CMP_TOLERANCE 0.0001f
#define CUDA_REDUCE_SAME_TYPE
#define IPP_ADDITIONAL_PARAMS
#define CL_ADDITIONAL_PARAMS
#define NPP_ADDITIONAL_PARAMS
#define MIDDLE_PARAM
#define NPP_MIDDLE_PARAM

#define BENCH_NAME Min
#define CV_OPERATION(src, dst) minMax(src, &dst[0])
#include "benchReduce.hpp"

#define BENCH_NAME Max
#define CV_OPERATION(src, dst) minMax(src, nullptr, &dst[0])
#include "benchReduce.hpp"

/*#define BENCH_NAME MinAbs
#include "benchReduce.hpp"

#define BENCH_NAME MaxAbs
#include "benchReduce.hpp"*/

#define BENCH_NAME Min
#include "benchReduce4C.hpp"

#define BENCH_NAME Max
#include "benchReduce4C.hpp"

#undef IPP_ADDITIONAL_PARAMS
#undef CL_ADDITIONAL_PARAMS
#undef NPP_ADDITIONAL_PARAMS
#define IPP_ADDITIONAL_PARAMS , (int*) &this->m_IndxIPP.X,  (int*) &this->m_IndxIPP.Y
#define CL_ADDITIONAL_PARAMS  , (int*) &this->m_IndxCL.X,   (int*) &this->m_IndxCL.Y
#define NPP_ADDITIONAL_PARAMS , (int*) &this->m_IndxNPP->X, (int*) &this->m_IndxNPP->Y

// NOTE : Coordinates returned are not currently checked
// custom image generation would be needed for a good check

#define BENCH_NAME MinIndx
#define CV_OPERATION(src, dst) minMaxLoc(src, &dst[0], &m_CVDummy[0], &m_IndxCV, &m_CVDummyIndx)
#include "benchReduce.hpp"

#define BENCH_NAME MaxIndx
#define CV_OPERATION(src, dst) minMaxLoc(src, &m_CVDummy[0], &dst[0], &m_CVDummyIndx, &m_IndxCV)
#include "benchReduce.hpp"

#undef IPP_REDUCE_HINT
#undef REDUCE_DST_TYPE
#undef CUDA_REDUCE_SAME_TYPE
#undef IPP_ADDITIONAL_PARAMS
#undef CL_ADDITIONAL_PARAMS
#undef NPP_ADDITIONAL_PARAMS

#define IPP_ADDITIONAL_PARAMS
#define CL_ADDITIONAL_PARAMS
#define NPP_ADDITIONAL_PARAMS

#define REDUCE_DST_TYPE double
#define IPP_REDUCE_HINT , ippAlgHintNone

#undef REDUCE_CMP_TOLERANCE
// Sum and Mean do calculations with very high values
// But in the GPU the calculations are done in floats
// So the results will have less precision
// So we allow higher tolerance here
#define REDUCE_CMP_TOLERANCE 0.001f

#define BENCH_NAME Sum
#define CV_OPERATION(src, dst) dst = sum(src)
#include "benchReduce.hpp"

#define BENCH_NAME Mean
#define CV_OPERATION(src, dst) { Scalar StdDev = 0; meanStdDev(src, dst, StdDev); }
#include "benchReduce.hpp"

/*#define BENCH_NAME MeanSqr
#include "benchReduce.hpp"*/

#define BENCH_NAME Sum
#include "benchReduce4C.hpp"

#define BENCH_NAME Mean
#include "benchReduce4C.hpp"

#undef  MIDDLE_PARAM
#undef  NPP_MIDDLE_PARAM
#undef  IPP_REDUCE_HINT
#define MIDDLE_PARAM &this->m_Dummy,
#define NPP_MIDDLE_PARAM this->m_NPPDummy,
#define IPP_REDUCE_HINT

#define BENCH_NAME Mean_StdDev
#define CV_OPERATION(src, dst) { Scalar Mean = 0; meanStdDev(src, Mean, dst); }
#include "benchReduce.hpp"

#undef MIDDLE_PARAM
#undef NPP_MIDDLE_PARAM
#undef IPP_REDUCE_HINT
#undef REDUCE_DST_TYPE
#undef REDUCE_CMP_TOLERANCE
