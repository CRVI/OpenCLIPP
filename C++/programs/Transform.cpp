////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.cpp
//! @date   : Jul 2013
//!
//! @brief  : Simple image transformation
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

#include "Programs/Transform.h"

#include "kernel_helpers.h"

#ifdef max
#undef max
#undef min
#endif

namespace OpenCLIPP
{

void Transform::MirrorX(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(mirror_x, Source, Dest);
}

void Transform::MirrorY(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(mirror_y, Source, Dest);
}

void Transform::Flip(IImage& Source, IImage& Dest)
{
   CheckCompatibility(Source, Dest);

   Kernel(flip, Source, Dest);
}

void Transform::Transpose(IImage& Source, IImage& Dest)
{
   if (Dest.Width() < Source.Height() || Dest.Height() < Source.Width())
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "Destination image too small to receive Transform::Transpose");

   if (Source.IsFloat() != Dest.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   Kernel(transpose, Source, Dest);
}

void Transform::Resize(IImage& Source, IImage& Dest, bool LinearInterpolation, bool KeepRatio)
{
   if (Source.IsFloat() != Dest.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   // This kernel works differently - we use the range of the destination
   // So we can't use the Kernel() macro

   const char * name = "resize";
   if (LinearInterpolation)
      name = "resize_linear";

   float RatioX = Source.Width() * 1.f / Dest.Width();
   float RatioY = Source.Height() * 1.f / Dest.Height();

   cl::NDRange Range = Dest.FullRange();

   if (KeepRatio)
   {
      float Ratio = std::max(RatioX, RatioY);
      RatioX = Ratio;
      RatioY = Ratio;

      Range = cl::NDRange(size_t(Source.Width() / RatioX), size_t(Source.Height() / RatioY));
   }

   auto kernel = cl::make_kernel<cl::Image2D, cl::Image2D, float, float>(SelectProgram(Source), name);
   kernel(cl::EnqueueArgs(*m_CL, Range), Source, Dest, RatioX, RatioY);
}

void Transform::SetAll(IImage& Dest, float Value)
{
   auto kernel = cl::make_kernel<cl::Image2D, float>(SelectProgram(Dest), "set_all");
   kernel(cl::EnqueueArgs(*m_CL, Dest.FullRange()), Dest, Value);
}

}
