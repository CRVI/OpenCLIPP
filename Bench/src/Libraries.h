////////////////////////////////////////////////////////////////////////////////
//! @file	: Libraries.h
//! @date   : Jul 2013
//!
//! @brief  : Includes headers of the selected image processing libraries
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

// OpenCLIPP
#include <OpenCLIPP.h>


#ifdef HAS_NPP
#include <npp.h>
#include <cufft.h>
#ifdef _MSC_VER
#pragma comment ( lib , "nppi" )
#pragma comment ( lib , "cuda" )
#pragma comment ( lib , "cudart" )
#pragma comment ( lib , "cufft" )
#endif _MSC_VER
#define NPP_AVAILABLE true
#define NPP_CODE(code) code
#else
#define NPP_AVAILABLE false
#define NPP_CODE(code)
#endif   // HAS_NPP

#ifdef HAS_IPP
#include <ipp.h>
#ifdef _MSC_VER
#pragma comment ( lib , "ippcore" )
#pragma comment ( lib , "ipps" )
#pragma comment ( lib , "ippi" )
#pragma comment ( lib , "ippcv" )
#pragma warning ( disable : 4996 )  // Disable deprecated warnings - we use IPP functions that are marked as deprecated but still supported
#endif _MSC_VER
#define IPP_AVAILABLE true
#define IPP_CODE(code) code
#else
#define IPP_AVAILABLE false
#define IPP_CODE(code)
#endif   // HAS_IPP

#ifdef HAS_CUDA
#include "custom_cuda.h"
#else
#define CUDA_AVAILABLE false
#define CUDA_CODE(code)
#endif   // HAS_CUDA

#ifdef HAS_CV
#include <opencv2/opencv.hpp>
#include <opencv2/ocl/ocl.hpp>
#ifdef _MSC_VER
#pragma comment ( lib , "opencv_core246" )
#pragma comment ( lib , "opencv_ocl246" )
#endif _MSC_VER
using namespace cv;
using namespace ocl;
#define CV_CODE(code) code
#define CV_AVAILABLE true
#else
#define CV_CODE(code)
#define CV_AVAILABLE false
#endif   // HAS_CV
