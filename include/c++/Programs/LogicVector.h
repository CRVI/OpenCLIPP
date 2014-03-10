////////////////////////////////////////////////////////////////////////////////
//! @file	: LogicVector.h
//! @date   : Jul 2013
//!
//! @brief  : Logic (bitwise) operations on image buffers
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

/// A program that does logic (bitwise) operations on image buffers
class CL_API LogicVector : public VectorProgram
{
public:
   LogicVector(COpenCL& CL)
      :  VectorProgram(CL, "Vector_Logic.cl")
   { }

   // Bitwise operations - float images not allowed
   void And(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);   ///< D = S1 & S2
   void Or(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);    ///< D = S1 | S2
   void Xor(ImageBuffer& Source1, ImageBuffer& Source2, ImageBuffer& Dest);   ///< D = S1 ^ S2
   void And(ImageBuffer& Source, ImageBuffer& Dest, uint value);              ///< D = S & v
   void Or(ImageBuffer& Source, ImageBuffer& Dest, uint value);               ///< D = S | v
   void Xor(ImageBuffer& Source, ImageBuffer& Dest, uint value);              ///< D = S ^ v
   void Not(ImageBuffer& Source, ImageBuffer& Dest);                          ///< D = ~S

};

}
