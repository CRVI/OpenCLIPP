////////////////////////////////////////////////////////////////////////////////
//! @file	: benchTranform.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates classes for all supported transformation reductions
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

// These are image only
#undef USE_BUFFER
#define USE_BUFFER false
#undef HAS_CL_BUFFER

#define HAS_FLOAT
#define CONSTANT_MIDDLE
#define CONSTANT_LAST
#define NO_CUSTOM_CUDA

#define BENCH_NAME MirrorX
#define IPP_NAME Mirror
#define CV_NAME flip
#define IPP_PARAM_LAST , ippAxsVertical
#define NPP_PARAM_LAST , NPP_VERTICAL_AXIS
#define CV_PARAM_LAST , 0
#include "benchUnary.hpp"

#define BENCH_NAME MirrorY
#define IPP_NAME Mirror
#define CV_NAME flip
#define IPP_PARAM_LAST , ippAxsHorizontal
#define NPP_PARAM_LAST , NPP_HORIZONTAL_AXIS
#define CV_PARAM_LAST , 1
#include "benchUnary.hpp"

#define BENCH_NAME Flip
#define IPP_NAME Mirror
#define CV_NAME flip
#define IPP_PARAM_LAST , ippAxsBoth
#define NPP_PARAM_LAST , NPP_BOTH_AXIS
#define CV_PARAM_LAST , -1
#include "benchUnary.hpp"

#undef NO_CUSTOM_CUDA

#define BENCH_NAME Transpose
#define CV_NAME transpose
#define ADDITIONNAL_DECLARATIONS \
   void Create(uint Width, uint Height)\
   {\
      IBench1in1out::Create<DataType, DataType>(Width, Height, Height, Width);\
   }

#include "benchUnary.hpp"
