////////////////////////////////////////////////////////////////////////////////
//! @file	: Logic.cpp
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

#include "Programs/Logic.h"


#define KERNEL_RANGE(...) GetRange(__VA_ARGS__)

#include "kernel_helpers.h"


namespace OpenCLIPP
{


// Bitwise operations - float images not allowed

void Logic::And(Image& Source1, Image& Source2,
                             Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(and_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Logic::Or(Image& Source1, Image& Source2,
                            Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(or_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Logic::Xor(Image& Source1, Image& Source2,
                             Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(xor_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Logic::And(Image& Source, Image& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(and_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void Logic::Or(Image& Source, Image& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(or_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void Logic::Xor(Image& Source, Image& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(xor_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void Logic::Not(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(not_image, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels());
}

}
