////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.cpp
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on images
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

#include "kernel_helpers.h"
#include "Programs/Statistics.h"
#include <vector>

namespace OpenCLIPP
{

   void ImageProximity::SqrDistance(IImage& Source, IImage& Template, IImage& Dest)
   {
      if (Template.Width() > Source.Width() || Template.Height() > Source.Height())
         throw cl::Error(CL_INVALID_VALUE, "Template image must be smaller than the source image.");

      CheckSameSize(Source, Dest);
      CheckFloat(Dest);

      Kernel(SqrDistance, In(Source, Template), Out(Dest), Source.Width(), Source.Height(), Template.Width(), Template.Height());
   }

}