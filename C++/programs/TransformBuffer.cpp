////////////////////////////////////////////////////////////////////////////////
//! @file	: TransformBuffer.cpp
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformation on image buffers
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

#include "Programs/TransformBuffer.h"


#define SELECT_NAME(name, src_img) SelectName( #name , src_img)

#include "kernel_helpers.h"

// Transpose uses a local array and a local size of 32*8
#define PIXELS_PER_WORKITEM_V   4
#define LOCAL_WIDTH  32
#define LOCAL_HEIGHT 8

#include "WorkGroup.h"


namespace OpenCLIPP
{

static std::string SelectName(const char * Name, const ImageBase& Img)
{
   if (Img.NbChannels() < 1 || Img.NbChannels() > 4)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Images must have between 1 and 4 channels");

   std::string KernelName = Name;
   KernelName += "_" + std::to_string(Img.NbChannels()) + "C";

   return KernelName;
}


void TransformBuffer::MirrorX(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_x, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::MirrorY(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_y, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::Flip(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(flip, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::Transpose(ImageBuffer& Source, ImageBuffer& Dest)
{
   if (Dest.Width() < Source.Height() || Dest.Height() < Source.Width())
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "Destination image too small to receive TransformBuffer::Transpose");

   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   if (IsFlushImage(Source))
   {
      Kernel_Local(transpose_flush, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
   }
   else
   {
      Kernel_Local(transpose, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
   }

}

void TransformBuffer::SetAll(ImageBuffer& Dest, float Value)
{
   Kernel(set_all, In(Dest), Out(), Dest.Step(), Value);
}

}
