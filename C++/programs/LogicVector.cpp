////////////////////////////////////////////////////////////////////////////////
//! @file	: LogicVector.cpp
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

#include "Programs/LogicVector.h"


#define KERNEL_RANGE(src_img) GetRange(src_img)

#include "kernel_helpers.h"


namespace OpenCLIPP
{


// Bitwise operations - float images not allowed

void LogicVector::And(ImageBuffer& Source1, ImageBuffer& Source2,
                             ImageBuffer& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(and_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void LogicVector::Or(ImageBuffer& Source1, ImageBuffer& Source2,
                            ImageBuffer& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(or_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void LogicVector::Xor(ImageBuffer& Source1, ImageBuffer& Source2,
                             ImageBuffer& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);
   CheckNotFloat(Source1);

   Kernel(xor_images, In(Source1, Source2), Out(Dest), Source1.Step(),
      Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void LogicVector::And(ImageBuffer& Source, ImageBuffer& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(and_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void LogicVector::Or(ImageBuffer& Source, ImageBuffer& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(or_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void LogicVector::Xor(ImageBuffer& Source, ImageBuffer& Dest, uint value)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(xor_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels(), value);
}

void LogicVector::Not(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);
   CheckNotFloat(Source);

   Kernel(not_image, In(Source), Out(Dest), Source.Step(), Dest.Step(),
      Source.Width() * Source.NbChannels());
}

}
