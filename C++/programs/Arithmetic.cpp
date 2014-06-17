////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.cpp
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations
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

#include "Programs/Arithmetic.h"


#define KERNEL_RANGE(...) GetRange(__VA_ARGS__)

#include "kernel_helpers.h"

namespace OpenCLIPP
{


// Between two images
void Arithmetic::Add(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(add_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::AddSquare(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(add_square_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Sub(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(sub_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::AbsDiff(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(abs_diff_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Mul(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(mul_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Div(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(div_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Min(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(min_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Max(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(max_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Mean(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(mean_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}

void Arithmetic::Combine(Image& Source1, Image& Source2, Image& Dest)
{
   CheckSimilarity(Source1, Source2);
   CheckSimilarity(Source1, Dest);

   Kernel(combine_images, In(Source1, Source2), Out(Dest), Source1.Step(), Source2.Step(), Dest.Step(), Source1.Width() * Source1.NbChannels());
}


// Image and value

void Arithmetic::Add(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(add_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Sub(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(sub_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::AbsDiff(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(abs_diff_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Mul(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(mul_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Div(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(div_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::RevDiv(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(reversed_div, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Min(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(min_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Max(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(max_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}

void Arithmetic::Mean(Image& Source, Image& Dest, float value)
{
   CheckSimilarity(Source, Dest);

   Kernel(mean_constant, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels(), value);
}


// Calculation on one image

void Arithmetic::Abs(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(abs_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Invert(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(invert_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Sqr(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(sqr_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}


// Calculation on one image - float required

void Arithmetic::Exp(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(exp_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Log(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(log_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Sqrt(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(sqrt_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Sin(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(sin_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

void Arithmetic::Cos(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(cos_image, In(Source), Out(Dest), Source.Step(), Dest.Step(), Source.Width() * Source.NbChannels());
}

}
