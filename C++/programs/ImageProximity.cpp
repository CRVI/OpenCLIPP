////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.cpp
//! @date   : Feb 2014
//!
//! @brief  : Image comparisons for pattern matching
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

#include "Programs/ImageProximity.h"

#include "WorkGroup.h"

#include "kernel_helpers.h"

namespace OpenCLIPP
{

void ImageProximity::SqrDistance(Image& Source, Image& Template, Image& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   if (Template.Width() <= 16 && Template.Height() <= 16)
   {
      Kernel_Local(SqrDistance_Cached, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
   else
   {
      Kernel(SqrDistance, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }

}

void ImageProximity::SqrDistance_Norm(Image& Source, Image& Template, Image& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   if (Template.Width() <= 16 && Template.Height() <= 16)
   {
      Kernel_Local(SqrDistance_Norm_Cached, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
   else
   {
      Kernel(SqrDistance_Norm, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
}

void ImageProximity::AbsDistance(Image& Source, Image& Template, Image& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   if (Template.Width() <= 16 && Template.Height() <= 16)
   {
      Kernel_Local(AbsDistance_Cached, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
   else
   {
      Kernel(AbsDistance, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
}

void ImageProximity::CrossCorr(Image& Source, Image& Template, Image& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   if (Template.Width() <= 16 && Template.Height() <= 16)
   {
      Kernel_Local(CrossCorr_Cached, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
   else
   {
      Kernel(CrossCorr, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
}

void ImageProximity::CrossCorr_Norm(Image& Source, Image& Template, Image& Dest)
{
   if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
      throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

   if(!SameType(Source, Template))
      throw cl::Error(CL_INVALID_VALUE, "Source and Template must have same type.");

   CheckSameSize(Source, Dest);
   CheckSameNbChannels(Source, Dest);
   CheckFloat(Dest);

   if (Template.Width() <= 16 && Template.Height() <= 16)
   {
      Kernel_Local(CrossCorr_Norm_Cached, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
   else
   {
      Kernel(CrossCorr_Norm, In(Source, Template), Out(Dest),
         Source.Step(), Template.Step(), Dest.Step(),
         Template.Width(), Template.Height(), Dest.Width(), Dest.Height());
   }
}

}
