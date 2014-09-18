////////////////////////////////////////////////////////////////////////////////
//! @file	: Integral.cpp
//! @date   : Mar 2014
//!
//! @brief  : Calculates the square integral sum scan of an image
//! 
//! Copyright (C) 2014 - CRVI
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


// With 16 and 16, too much local memory is used so we use 8 and 8
#define LOCAL_WIDTH 8
#define PIXELS_PER_WORKITEM_H 8

#include "WorkGroup.h"


#include "kernel_helpers.h"


using namespace cl;

namespace OpenCLIPP
{
void Integral::PrepareFor(ImageBase& Source)
{
   ImageProgram::PrepareFor(Source);

   SSize VerticalImgSize_F32 = {GetNbGroupsW(Source) - 1, Source.Height()};
   SSize HorizontalImgSize_F32 = {Source.Width(), GetNbGroupsH(Source) - 1};

   if (VerticalImgSize_F32.Width == 0)
      VerticalImgSize_F32.Width = 1;

   if (HorizontalImgSize_F32.Height == 0)
      HorizontalImgSize_F32.Height = 1;

   // Check validity of current temp buffers
   if (m_VerticalJunctions_F32 != nullptr && 
      uint(VerticalImgSize_F32.Width) <= m_VerticalJunctions_F32->Width() &&
      uint(VerticalImgSize_F32.Height) <= m_VerticalJunctions_F32->Height() &&
      uint(HorizontalImgSize_F32.Width) <= m_HorizontalJunctions_F32->Width() &&
      uint(HorizontalImgSize_F32.Height) <= m_HorizontalJunctions_F32->Height())
   {
      // Buffers are good
      return;
   }

   // Create buffers for temporary results
   m_VerticalJunctions_F32 = std::make_shared<TempImage>(*m_CL, VerticalImgSize_F32, SImage::F32);
   m_HorizontalJunctions_F32 = std::make_shared<TempImage>(*m_CL, HorizontalImgSize_F32, SImage::F32);

   //-------------------------------------------------------------------------------------
   SSize VerticalImgSize_F64 = {GetNbGroupsW(Source) - 1, Source.Height()};
   SSize HorizontalImgSize_F64 = {Source.Width(), GetNbGroupsH(Source) - 1};

   if (VerticalImgSize_F64.Width == 0)
      VerticalImgSize_F64.Width = 1;

   if (HorizontalImgSize_F64.Height == 0)
      HorizontalImgSize_F64.Height = 1;

   // Check validity of current temp buffers
   if (m_VerticalJunctions_F64 != nullptr && 
      uint(VerticalImgSize_F64.Width) <= m_VerticalJunctions_F64->Width() &&
      uint(VerticalImgSize_F64.Height) <= m_VerticalJunctions_F64->Height() &&
      uint(HorizontalImgSize_F64.Width) <= m_HorizontalJunctions_F64->Width() &&
      uint(HorizontalImgSize_F64.Height) <= m_HorizontalJunctions_F64->Height() &&
      Source.IsFloat() == m_VerticalJunctions_F64->IsFloat())
   {
      // Buffers are good
      return;
   }

   // Create buffers for temporary results
   m_VerticalJunctions_F64 = std::make_shared<TempImage>(*m_CL, VerticalImgSize_F64, SImage::F64);
   m_HorizontalJunctions_F64 = std::make_shared<TempImage>(*m_CL, HorizontalImgSize_F64, SImage::F64);
}

void Integral::IntegralSum(Image& Source, Image& Dest)
{
   switch (Dest.DataType())
   {
   case SImage::F32:
      Integral_F32(Source, Dest);
      break;
   case SImage::F64:
      Integral_F64(Source, Dest);
      break;
   case SImage::U8:
   case SImage::S8:
   case SImage::U16:
   case SImage::S16:
   case SImage::U32:
   case SImage::S32:
   case SImage::NbDataTypes:
   default:
      throw cl::Error(CL_INVALID_VALUE, "Destination image of Integral must be F32 or F64");
   }

}

void Integral::SqrIntegral(Image& Source, Image& Dest)
{
   switch (Dest.DataType())
   {
   case SImage::F32:
      SqrIntegral_F32(Source, Dest);
      break;
   case SImage::F64:
      SqrIntegral_F64(Source, Dest);
      break;
   case SImage::U8:
   case SImage::S8:
   case SImage::U16:
   case SImage::S16:
   case SImage::U32:
   case SImage::S32:
   case SImage::NbDataTypes:
   default:
      throw cl::Error(CL_INVALID_VALUE, "Destination image of Integral must be F32 or F64");
   }

}

void Integral::Integral_F32(Image& Source, Image& Dest)
{
   PrepareFor(Source);
   CheckSameSize(Source, Dest);

   Kernel_Local(scan1_F32, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan2_F32")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions_F32, Dest.Step(), m_VerticalJunctions_F32->Step());
   }

   Kernel(scan3_F32, In(Dest, *m_VerticalJunctions_F32), Dest, Dest.Step(), m_VerticalJunctions_F32->Step(), Dest.Step());

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan4_F32")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions_F32, Dest.Step(), m_HorizontalJunctions_F32->Step());
   }

   Kernel(scan5_F32, In(Dest, *m_HorizontalJunctions_F32), Dest, Dest.Step(), m_HorizontalJunctions_F32->Step(), Dest.Step());
   
}

void Integral::SqrIntegral_F32(Image& Source, Image& Dest)
{
   PrepareFor(Source);
   CheckSameSize(Source, Dest);

   Kernel_Local(sqr_F32, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan2_F32")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions_F32, Dest.Step(), m_VerticalJunctions_F32->Step());
   }

   Kernel(scan3_F32, In(Dest, *m_VerticalJunctions_F32), Dest, Dest.Step(), m_VerticalJunctions_F32->Step(), Dest.Step());

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan4_F32")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions_F32, Dest.Step(), m_HorizontalJunctions_F32->Step());
   }

   Kernel(scan5_F32, In(Dest, *m_HorizontalJunctions_F32), Dest, Dest.Step(), m_HorizontalJunctions_F32->Step(), Dest.Step());
   
}

void Integral::Integral_F64(Image& Source, Image& Dest)
{
   PrepareFor(Source);
   CheckSameSize(Source, Dest);

   Kernel_Local(scan1_F64, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan2_F64")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions_F64, Dest.Step(), m_VerticalJunctions_F64->Step());
   }

   Kernel(scan3_F64, In(Dest, *m_VerticalJunctions_F64), Dest, Dest.Step(), m_VerticalJunctions_F64->Step(), Dest.Step());

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan4_F64")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions_F64, Dest.Step(), m_HorizontalJunctions_F64->Step());
   }

   Kernel(scan5_F64, In(Dest, *m_HorizontalJunctions_F64), Dest, Dest.Step(), m_HorizontalJunctions_F64->Step(), Dest.Step());
   
}

void Integral::SqrIntegral_F64(Image& Source, Image& Dest)
{
   PrepareFor(Source);
   CheckSameSize(Source, Dest);

   Kernel_Local(sqr_F64, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());

   if (GetNbGroupsW(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan2_F64")
         (EnqueueArgs(*m_CL, NDRange(GetNbGroupsW(Source) - 1, Source.Height(), 1)), Dest, *m_VerticalJunctions_F64, Dest.Step(), m_VerticalJunctions_F64->Step());
   }

   Kernel(scan3_F64, In(Dest, *m_VerticalJunctions_F64), Dest, Dest.Step(), m_VerticalJunctions_F64->Step(), Dest.Step());

   if (GetNbGroupsH(Source) > 1)
   {
      make_kernel<cl::Buffer, cl::Buffer, uint, uint>(SelectProgram(Dest), "scan4_F64")
         (EnqueueArgs(*m_CL, NDRange(Source.Width(), GetNbGroupsH(Source) - 1, 1)), Dest, *m_HorizontalJunctions_F64, Dest.Step(), m_HorizontalJunctions_F64->Step());
   }

   Kernel(scan5_F64, In(Dest, *m_HorizontalJunctions_F64), Dest, Dest.Step(), m_HorizontalJunctions_F64->Step(), Dest.Step());
   
}

}
