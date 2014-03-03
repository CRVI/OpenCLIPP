////////////////////////////////////////////////////////////////////////////////
//! @file	: Basic.h 
//! @date   : Jul 2013
//!
//! @brief  : General declarations needed by the library
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

#pragma once

#ifdef _MSC_VER

   #ifdef CL_EXPORTS
   #define CL_API __declspec(dllexport)
   #else
   #define CL_API __declspec(dllimport)
   #endif

   // cl.hpp needs a higher value for this (default value is 5)
   #define _VARIADIC_MAX 10

#else   // _MSC_VER

#define CL_API

#endif  // _MSC_VER


#include <assert.h>

typedef unsigned int uint;

//#define USE_CLFFT  // Uncomment to enable using clFFT library to compute Fast Fourrier Transform
