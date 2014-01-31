////////////////////////////////////////////////////////////////////////////////
//! @file	: config.h
//! @date   : Jul 2013
//!
//! @brief  : Configuration of benchmark program
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

/// NOTE : These config should be changed to be command line parameters

//Config
//#define FULL_BENCH    // All image sizes - comment to run only a single image size for faster run time
//#define FULL_TESTS      // All tests - uncomment to run unit tests instead of benchmark
#define WAITFORKEY_AT_END

#define SUCCESS_EPSILON 0.0001f
