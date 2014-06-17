////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.cl
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

#define ARG_TYPE int

#include "Vector.h"


// Bitwise operations - float not allowed
#ifndef FLOAT

BINARY_OP(and_images, src1 & src2)
BINARY_OP(or_images, src1 | src2)
BINARY_OP(xor_images, src1 ^ src2)
// image and value
CONSTANT_OP(and_constant, src & value)
CONSTANT_OP(or_constant, src | value)
CONSTANT_OP(xor_constant, src ^ value)

// Unary
UNARY_OP(not_image, ~src)

#endif   // not FLOAT
