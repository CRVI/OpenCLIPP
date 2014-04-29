////////////////////////////////////////////////////////////////////////////////
//! @file	: benchImageProximityBase.hpp
//! @date   : Mar 2014
//!
//! @brief  : Base class for Image Proximity operations
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

#include <memory>

#define PROXIMITY_USE_BUFFER USE_BUFFER

#define BENCH_NAME SqrDistance_Norm
#define IPP_NAME SqrDistanceSame_Norm
#include "benchImageProximity.hpp"

/* // These are not supported in IPP
#define BENCH_NAME SqrDistance    
#include "benchImageProximity.hpp"

#define BENCH_NAME AbsDistance
#include "benchImageProximity.hpp"

#define BENCH_NAME CrossCorr
#include "benchImageProximity.hpp"*/

#define BENCH_NAME CrossCorr_Norm
#define IPP_NAME CrossCorrSame_Norm
#include "benchImageProximity.hpp"