////////////////////////////////////////////////////////////////////////////////
//! @file	: Buffers.h
//! @date   : Mar 2014
//!
//! @brief  : Macros for accessing images stored in OpenCL buffers
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

#include "Base.h"


#ifdef S8
#define SCALAR char
#endif

#ifdef U8
#define SCALAR uchar
#endif

#ifdef S16
#define SCALAR short
#endif

#ifdef U16
#define SCALAR ushort
#endif

#ifdef S32
#define SCALAR int
#endif

#ifdef U32
#define SCALAR uint
#endif

#ifdef F32
#define SCALAR float
#define FLOAT
#endif

#ifdef F64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define SCALAR double
#define FLOAT
#endif

#ifndef SCALAR
#define SCALAR uchar
#endif

#define INPUT_SPACE global    // If input images are read only, they can be set to be in "constant" memory space, with possible speed improvements



#ifdef FLOAT
#define CONVERT_SCALAR(val) val
#define ABS fabs
#else
#define CONVERT_SCALAR(val) CONCATENATE(CONCATENATE(convert_, SCALAR), _sat) (val)  // Example : convert_uchar_sat(val)
#define ABS abs
#endif

#define TYPE2 CONCATENATE(SCALAR, 2)
#define TYPE3 CONCATENATE(SCALAR, 3)
#define TYPE4 CONCATENATE(SCALAR, 4)
