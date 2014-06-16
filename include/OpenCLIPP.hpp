////////////////////////////////////////////////////////////////////////////////
//! @file	: OpenCLIPP.hpp
//! @date   : Jan 2014
//!
//! @brief  : Main include file for C++ interface of OpenCLIPP
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

#include "c++/OpenCL.h"
#include "c++/Buffer.h"
#include "c++/Image.h"
#include "c++/Programs/Program.h"
#include "c++/Programs/Arithmetic.h"
#include "c++/Programs/ArithmeticVector.h"
#include "c++/Programs/Blob.h"
#include "c++/Programs/Conversions.h"
#include "c++/Programs/ConversionsBuffer.h"
#include "c++/Programs/Filters.h"
#include "c++/Programs/FiltersVector.h"
#include "c++/Programs/Histogram.h"
#include "c++/Programs/Integral.h"
#include "c++/Programs/IntegralBuffer.h"
#include "c++/Programs/Logic.h"
#include "c++/Programs/LogicVector.h"
#include "c++/Programs/Lut.h"
#include "c++/Programs/LutVector.h"
#include "c++/Programs/Morphology.h"
#include "c++/Programs/MorphologyBuffer.h"
#include "c++/Programs/ImageProximity.h"
#include "c++/Programs/ImageProximityBuffer.h"
#include "c++/Programs/ImageProximityFFT.h"
#include "c++/Programs/Statistics.h"
#include "c++/Programs/StatisticsVector.h"
#include "c++/Programs/Transform.h"
#include "c++/Programs/TransformBuffer.h"
#include "c++/Programs/Thresholding.h"
#include "c++/Programs/ThresholdingVector.h"
#include "c++/Programs/FFT.h"
