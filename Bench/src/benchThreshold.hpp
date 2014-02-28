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

#define THRESHOLD_TYPE GT
#define BENCH_NAME Threshold
#include "benchThresholdBase.hpp"

#define THRESHOLD_TYPE LT
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