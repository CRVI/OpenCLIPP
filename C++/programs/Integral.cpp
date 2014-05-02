////////////////////////////////////////////////////////////////////////////////
//! @file	: Integral.cpp
//! @date   : Jul 2013
//!
//! @brief  : Calculates the integral sum scan of an image
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

#include "Programs/Integral.h"


#define PIXELS_PER_WORKITEM_H 16

#include "WorkGroup.h"


#include "kernel_helpers.h"


using namespace cl;

namespace OpenCLIPP
{

void Integral::PrepareFor(ImageBase& Source)
{
   ImageProgram::PrepareFor(Source);

   // Also build float program as we will need it
   GetProgram(Float).Build();

   SSize VerticalImgSize = {GetNbGroupsW(Source) - 1, Source.Height()};
   SSize HorizontalImgSize = {Source.Width(), GetNbGroupsH(Source) - 1};

   if (VerticalImgSize.Width == 0)
      VerticalImgSize.Width = 1;

   if (HorizontalImgSize.Height == 0)
      HorizontalImgSize.Height = 1;

   // Check validity of current temp buffers
   if (m_VerticalJunctions != nullptr && 
      uint(VerticalImgSize.Width) <= m_VerticalJunctions->Width() &&
      uint(VerticalImgSize.Height) <= m_VerticalJunctions->Height() &&
      uint(HorizontalImgSize.Width) <= m_HorizontalJunctions->Width() &&
      uint(HorizontalImgSize.Height) <= m_HorizontalJunctions->Height() &&
      Source.IsFloat() == m_VerticalJunctions->IsFloat())
   {
      // Buffers are good
      return;
   }

   // Create buffers for temporary results
   m_VerticalJunctions = std::make_shared<TempImage>(*m_CL, VerticalImgSize, SImage::F32);
   m_HorizontalJunctions = std::make_shared<TempImage>(*m_CL, HorizontalImgSize, SImage::F32);
}

void Integral::IntegralSum(IImage& Source, IImage& Dest)
{
   PrepareFor(Source);

   CheckSameSize(Source, Dest);
   CheckFloat(Dest);

   if (Dest.DataType() != SImage::F32)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of SqrIntegral must be float type");

   uint Width = Source.Width();
   uint Height = Source.Height();

   Kernel_Local(scan1, Source, Dest, Width, Height);

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<Image2D, Image2D>(SelectProgram(Dest), "scan2")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions);
   }

   Kernel(scan3, In(Dest, *m_VerticalJunctions), Dest);

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<Image2D, Image2D>(SelectProgram(Dest), "scan4")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions);
   }

   Kernel(scan5, In(Dest, *m_HorizontalJunctions), Dest);
}

void Integral::SqrIntegral(IImage& Source, IImage& Dest)
{
   PrepareFor(Source);

   CheckSameSize(Source, Dest);
   CheckFloat(Dest);

   if (Dest.DataType() != SImage::F32)
     throw cl::Error(CL_INVALID_VALUE, "Destination image of SqrIntegral must be float type");

   uint Width = Source.Width();
   uint Height = Source.Height();

   Kernel_Local(sqr, Source, Dest, Width, Height);

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<Image2D, Image2D>(SelectProgram(Dest), "scan2")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions);
   }

   Kernel(scan3, In(Dest, *m_VerticalJunctions), Dest);

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<Image2D, Image2D>(SelectProgram(Dest), "scan4")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions);
   }

   Kernel(scan5, In(Dest, *m_HorizontalJunctions), Dest);
}

}
