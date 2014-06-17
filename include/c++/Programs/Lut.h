////////////////////////////////////////////////////////////////////////////////
//! @file	: Lut.h
//! @date   : Jul 2013
//!
//! @brief  : LUT transformation of images
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

#include "Program.h"

namespace OpenCLIPP
{

/// A program that does LUT (Look Up Table) transformation of images
class CL_API Lut : public ImageProgram
{
public:
   Lut(COpenCL& CL)
   :  ImageProgram(CL, "Lut.cl")
   { }

   /// Performs a LUT operation.
   /// levels and values must be arrays of NbValues elements
   /// Dest will contain the following transformation :
   /// find value v where (S >= levels[v] && S < levels[v + 1])
   /// D = values[v]
   /// \param levels : Contains NbValues describing the levels to look at in Source
   /// \param values : Contains NbValues describing the values to use for those levels
   void LUT(Image& Source, Image& Dest, uint * levels, uint * values, uint NbValues);

   /// Performs a linear LUT operation.
   /// levels and values must be arrays of NbValues elements
   /// Dest will contain the following transformation :
   /// find value v where (S >= levels[v] && S < levels[v + 1])
   /// ratio = (S - levels[v]) / (levels[v + 1] - levels[v])
   /// D = values[v] + (values[v + 1] - values[v]) * ratio
   /// \param levels : Contains NbValues describing the levels to look at in Source
   /// \param values : Contains NbValues describing the values to use for those levels
   void LUTLinear(Image& Source, Image& Dest, float * levels, float * values, uint NbValues);

   /// Performs a LUT on 8 bit unsigned images.
   /// D = values[S]
   void BasicLut(Image& Source, Image& Dest, unsigned char * values);

   /// Scales values of Source image according to the given input and output ranges
   void Scale(Image& Source, Image& Dest, float SrcMin, float SrcMax, float DstMin, float DstMax);
};

}
