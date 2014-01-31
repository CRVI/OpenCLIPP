////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.h
//! @date   : Jul 2013
//!
//! @brief  : Logic (bitwise) operations on images
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

/// A program that does logic (bitwise) operations on images
class CL_API Logic : public ImageProgram
{
public:
   Logic(COpenCL& CL)
   :  ImageProgram(CL, "Logic.cl")
   { }

   // Bitwise operations
   void And(IImage& Source1, IImage& Source2, IImage& Dest);   ///< D = S1 & S2
   void Or(IImage& Source1, IImage& Source2, IImage& Dest);    ///< D = S1 | S2
   void Xor(IImage& Source1, IImage& Source2, IImage& Dest);   ///< D = S1 ^ S2
   void And(IImage& Source, IImage& Dest, uint value);           ///< D = S & v
   void Or(IImage& Source, IImage& Dest, uint value);            ///< D = S | v
   void Xor(IImage& Source, IImage& Dest, uint value);           ///< D = S ^ v
   void Not(IImage& Source, IImage& Dest);                       ///< D = ~S

};

}
