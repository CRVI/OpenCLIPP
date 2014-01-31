////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.cpp
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

#include "Programs/Logic.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{


// Bitwise operations

void Logic::And(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(and_images, In(Source1, Source2), Out(Dest));
}

void Logic::Or(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(or_images, In(Source1, Source2), Out(Dest));
}

void Logic::Xor(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(xor_images, In(Source1, Source2), Out(Dest));
}

void Logic::And(IImage& Source, IImage& Dest, uint value)
{
   CheckCompatibility(Source, Dest);
   CheckNotFloat(Source);

   Kernel(and_constant, In(Source), Out(Dest), value);
}

void Logic::Or(IImage& Source, IImage& Dest, uint value)
{
   CheckCompatibility(Source, Dest);
   CheckNotFloat(Source);

   Kernel(or_constant, In(Source), Out(Dest), value);
}

void Logic::Xor(IImage& Source, IImage& Dest, uint value)
{
   CheckCompatibility(Source, Dest);
   CheckNotFloat(Source);

   Kernel(xor_constant, In(Source), Out(Dest), value);
}

void Logic::Not(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);
   CheckNotFloat(Source);

   Kernel(not_image, In(Source), Out(Dest));
}

}
