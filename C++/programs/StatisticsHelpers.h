////////////////////////////////////////////////////////////////////////////////
//! @file	: StatisticsHelpers.h
//! @date   : Jul 2013
//!
//! @brief  : Tools to help implementation of reduction-type programs
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

// Helpers for Statistics reduction

// Local group & global group logic :
//  The reduction algorithm expects to have each local group filled with workers (full grids of 16x16)
//  Each worker reads 16 pixels along X (so one group reads a rectangle of 16x16x16 (4096) pixels)
//  Local size is 16x16
//  So for images that have a Width not a multiple of 16*16 (256) or a height not a multiple of 16 :
//    Additional workers are started that only set in_image[lid] to false
//  For images with flush width and height, the faster _flush version is used
//  Each group will work on at least 1 pixel (worst case), so m_Result will be filled with valid data

#pragma once

#include "Image.h"


#define PIXELS_PER_WORKITEM_H 16

#include "WorkGroup.h"

namespace OpenCLIPP
{

std::string SelectName(const char * name, const ImageBase& Img);    // Selects the faster flush kernel if image is flush

// CPU side final reduction - done in double for higher precision
double ReduceSum(std::vector<float>& buffer);
double ReduceMean(std::vector<float>& buffer);

void ReduceSum(std::vector<float>& buffer, int NbChannels, double outVal[4]);
void ReduceMean(std::vector<float>& buffer, int NbChannels, double outVal[4]);

double ReduceMin(std::vector<float>& buffer, std::vector<int>& coords, int& outX, int& outY);
double ReduceMax(std::vector<float>& buffer, std::vector<int>& coords, int& outX, int& outY);

}
