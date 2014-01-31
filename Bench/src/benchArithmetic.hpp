////////////////////////////////////////////////////////////////////////////////
//! @file	: benchArithmetic.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a benchmark class for every supported Arithmetic primitives
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

/*
Defines used in this file :

USE_BUFFER :
   Needs to be set before including this file
   Set to true or false
   Used to select Image Buffer or Image mode of ocip for primitives that support both

BENCH_NAME :
   Name of the primitive to test

CONSTANT_LAST :
   Constant value given as last parameter to some IPP and NPP functions

CV_NAME :
   Name of the function in OpenCV OCL that is equivalent to the primitive

CONSTANT_MIDDLE :
   Constant value given as third parameter to some IPP and NPP functions

HAS_FLOAT :
   Defined for primitives that can accept float (F32) images

NO_UNSIGNED :
   Defined for primitives that can't accept unsigned images

DIVIDE_TYPE_TOLERANCE :
   Indicates that the results may differ due to rounding differences so tolerance is increased
*/


#define CONSTANT_LAST
#define CONSTANT_MIDDLE
#define HAS_FLOAT

#define BENCH_NAME Add
#define CV_NAME add
#include "benchArithmeticBinary.hpp"

/*#define BENCH_NAME AddSquare
#include "benchArithmeticBinary.hpp"
#undef BENCH_NAME*/

#define BENCH_NAME Sub
#define CV_NAME subtract
#include "benchArithmeticBinary.hpp"

#define BENCH_NAME AbsDiff
#define CV_NAME absdiff
#include "benchBinary.hpp"

#define BENCH_NAME Mul
#define CV_NAME multiply
#include "benchArithmeticBinary.hpp"

#define BENCH_NAME Div
#define CV_NAME divide
#define DIVIDE_TYPE_TOLERANCE
#include "benchArithmeticBinary.hpp"
#undef DIVIDE_TYPE_TOLERANCE

/*#define BENCH_NAME ImgMin
#include "benchArithmeticBinary.hpp"
#undef BENCH_NAME

#define BENCH_NAME ImgMax
#include "benchArithmeticBinary.hpp"
#undef BENCH_NAME

#define BENCH_NAME ImgMean
#include "benchArithmeticBinary.hpp"
#undef BENCH_NAME

#define BENCH_NAME Combine
#include "benchArithmeticBinary.hpp"
#undef BENCH_NAME*/


#undef CONSTANT_LAST
#define CONSTANT_LAST , 7

#define BENCH_NAME AddC
#define CV_NAME add
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME SubC
#define CV_NAME subtract
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME AbsDiffC
#define CV_NAME absdiff
#include "benchUnary.hpp"

#define BENCH_NAME MulC
#define CV_NAME multiply
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME DivC
#define CV_NAME divide
#define DIVIDE_TYPE_TOLERANCE
#include "benchArithmeticUnary.hpp"
#undef DIVIDE_TYPE_TOLERANCE


/* IPP & NPP do not have these
#define BENCH_NAME RevDivC
#include "benchUnary.hpp"
#undef BENCH_NAME

#define BENCH_NAME MinC
#include "benchUnary.hpp"
#undef BENCH_NAME

#define BENCH_NAME MaxC
#include "benchUnary.hpp"
#undef BENCH_NAME

#define BENCH_NAME MeanC
#include "benchUnary.hpp"
#undef BENCH_NAME*/


#undef CONSTANT_LAST
#define CONSTANT_LAST

#define BENCH_NAME Abs
//#define CV_NAME ocl::abs    // Listed in the documentation but not actually in OpenCV OCL
#define NO_UNSIGNED
#include "benchUnary.hpp"
#undef NO_UNSIGNED

#define BENCH_NAME Exp
#define CV_NAME exp
#define DIVIDE_TYPE_TOLERANCE
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME Sqrt
// OpenCV OCL has no Sqrt
#define NO_NEGATIVE
#include "benchArithmeticUnary.hpp"
#undef NO_NEGATIVE
#undef DIVIDE_TYPE_TOLERANCE

#define BENCH_NAME Sqr
// OpenCV OCL has no Sqr, has pow(a, 2) instead
#include "benchArithmeticUnary.hpp"

#if !defined(HAS_IPP) && !defined(HAS_NPP)

#define BENCH_NAME Sin
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME Cos
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME Log
#include "benchArithmeticUnary.hpp"

#define BENCH_NAME Invert
#include "benchUnary.hpp"

#endif   // Not IPP nor NPP



// Logic

#undef HAS_FLOAT

#define BENCH_NAME And
#define CV_NAME bitwise_and
#include "benchBinary.hpp"

#define BENCH_NAME Or
#define CV_NAME bitwise_or
#include "benchBinary.hpp"

#define BENCH_NAME Xor
#define CV_NAME bitwise_xor
#include "benchBinary.hpp"

// Logic + constant
#undef CONSTANT_MIDDLE
#define CONSTANT_MIDDLE , 7

#define BENCH_NAME AndC
#define CV_NAME bitwise_and
#include "benchUnary.hpp"

#define BENCH_NAME OrC
#define CV_NAME bitwise_or
#include "benchUnary.hpp"

#define BENCH_NAME XorC
#define CV_NAME bitwise_xor
#include "benchUnary.hpp"

// Logic unary
#undef CONSTANT_MIDDLE
#define CONSTANT_MIDDLE

/*#define BENCH_NAME Not   // Not supported in 16u
#include "benchUnary.hpp"*/

#undef CONSTANT_LAST
#undef CONSTANT_MIDDLE
