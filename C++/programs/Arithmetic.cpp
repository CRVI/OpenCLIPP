////////////////////////////////////////////////////////////////////////////////
//! @file	: Arithmetic.cpp
//! @date   : Jul 2013
//!
//! @brief  : Arithmetic operations on images
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

#include "kernel_helpers.h"


namespace OpenCLIPP
{

// Between two images

void Arithmetic::Add(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(add_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::AddSquare(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(add_square_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Sub(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(sub_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::AbsDiff(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(abs_diff_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Mul(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(mul_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Div(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(div_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Min(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(min_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Max(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(max_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Mean(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(mean_images, In(Source1, Source2), Out(Dest));
}

void Arithmetic::Combine(IImage& Source1, IImage& Source2, IImage& Dest)
{
   CheckCompatibility(Source1, Source2);
   CheckCompatibility(Source1, Dest);

   Kernel(combine, In(Source1, Source2), Out(Dest));
}


// Image and value

void Arithmetic::Add(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(add_constant, In(Source), Out(Dest), value);
}

void Arithmetic::Sub(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(sub_constant, In(Source), Out(Dest), value);
}

void Arithmetic::AbsDiff(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(abs_diff_constant, In(Source), Out(Dest), value);
}

void Arithmetic::Mul(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(mul_constant, In(Source), Out(Dest), value);
}

void Arithmetic::Div(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(div_constant, In(Source), Out(Dest), value);
}

void Arithmetic::RevDiv(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(reversed_div, In(Source), Out(Dest), value);
}

void Arithmetic::Min(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(min_constant, In(Source), Out(Dest), value);
}

void Arithmetic::Max(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(max_constant, In(Source), Out(Dest), value);
}

void Arithmetic::Mean(IImage& Source, IImage& Dest, float value)
{
   CheckCompatibility(Source, Dest);

   Kernel(mean_constant, In(Source), Out(Dest), value);
}


// Calculation on one image

void Arithmetic::Exp(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(exp_image, In(Source), Out(Dest));
}

void Arithmetic::Log(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(log_image, In(Source), Out(Dest));
}

void Arithmetic::Sqr(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(sqr_image, In(Source), Out(Dest));
}

void Arithmetic::Sqrt(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(sqrt_image, In(Source), Out(Dest));
}

void Arithmetic::Sin(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(sin_image, In(Source), Out(Dest));
}

void Arithmetic::Cos(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(cos_image, In(Source), Out(Dest));
}

void Arithmetic::Abs(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(abs_image, In(Source), Out(Dest));
}

void Arithmetic::Invert(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(invert_image, In(Source), Out(Dest));
}

}
