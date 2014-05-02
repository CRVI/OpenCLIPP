////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximityBuffer.cpp
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on image buffers
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

#include "Programs/ImageProximityBuffer.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void ImageProximityBuffer::SqrDistance(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   Kernel(SqrDistance, In(Source, Template), Out(Dest),
      Source.Step(), Template.Step(), Dest.Step(),
      Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
}

void ImageProximityBuffer::SqrDistance_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   Kernel(SqrDistance_Norm, In(Source, Template), Out(Dest),
      Source.Step(), Template.Step(), Dest.Step(),
      Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
}

void ImageProximityBuffer::AbsDistance(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   Kernel(AbsDistance, In(Source, Template), Out(Dest),
      Source.Step(), Template.Step(), Dest.Step(),
      Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
}

void ImageProximityBuffer::CrossCorr(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   Kernel(CrossCorr, In(Source, Template), Out(Dest),
      Source.Step(), Template.Step(), Dest.Step(),
      Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
}

void ImageProximityBuffer::CrossCorr_Norm(ImageBuffer& Source, ImageBuffer& Template, ImageBuffer& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   Kernel(CrossCorr_Norm, In(Source, Template), Out(Dest),
      Source.Step(), Template.Step(), Dest.Step(),
      Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
}

}
