////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.h
//! @date   : Jul 2013
//!
//! @brief  : Logic (bitwise) operations
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

/// A program that does logic (bitwise) operations
class CL_API Logic : public VectorProgram
{
public:
   Logic(COpenCL& CL)
      :  VectorProgram(CL, "Logic.cl")
   { }

   // Bitwise operations - float images not allowed
   void And(Image& Source1, Image& Source2, Image& Dest);   ///< D = S1 & S2
   void Or(Image& Source1, Image& Source2, Image& Dest);    ///< D = S1 | S2
   void Xor(Image& Source1, Image& Source2, Image& Dest);   ///< D = S1 ^ S2
   void And(Image& Source, Image& Dest, uint value);              ///< D = S & v
   void Or(Image& Source, Image& Dest, uint value);               ///< D = S | v
   void Xor(Image& Source, Image& Dest, uint value);              ///< D = S ^ v
   void Not(Image& Source, Image& Dest);                          ///< D = ~S

};

}
