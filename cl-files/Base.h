////////////////////////////////////////////////////////////////////////////////
//! @file	: Base.h
//! @date   : Mar 2014
//!
//! @brief  : Basic declarations
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

#ifdef __NV_CL_C_VERSION
#define NVIDIA_PLATFORM
#endif

#ifdef _AMD_OPENCL
#define AMD_PLATFORM
#endif

#ifdef NVIDIA_PLATFORM
   #define CONST static constant const
   #define CONST_ARG constant const
#else
   #ifdef AMD_PLATFORM
      #define CONST const
      #define CONST_ARG const
   #else
      #define CONST constant const
      #define CONST_ARG constant const
   #endif
#endif

#define CONCATENATE(a, b) _CONCATENATE(a, b)
#define _CONCATENATE(a, b) a ## b
