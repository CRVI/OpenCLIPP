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


#define VALUE 100
#define USHORT_VALUE 5000
#define FLOAT_VALUE 0.5

#define COMPARE_TYPE LT
#include "benchCompareBase.hpp"

#define COMPARE_TYPE LQ
#include "benchCompareBase.hpp"

#define COMPARE_TYPE EQ
#include "benchCompareBase.hpp"

#define COMPARE_TYPE GQ
#include "benchCompareBase.hpp"

#define COMPARE_TYPE GT
#include "benchCompareBase.hpp"


#define COMPARE_TYPE LT
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE LQ
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE EQ
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE GQ
#include "benchCompareImgBase.hpp"

#define COMPARE_TYPE GT
#include "benchCompareImgBase.hpp"
