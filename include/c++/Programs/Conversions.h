////////////////////////////////////////////////////////////////////////////////
//! @file	: Conversions.h
//! @date   : Jul 2013
//!
//! @brief  : Image depth conversion
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

#pragma once

#include "Program.h"

namespace OpenCLIPP
{

/// A program used to Convert the type of an image
class CL_API Conversions : public ImageProgram
{
public:
   Conversions(COpenCL& CL)
   :  ImageProgram(CL, "Convert.cl")
   { }

   /// From any image type to any image type - no value scaling
   void Convert(IImage& Source, IImage& Dest);

   /// From any image type to any image type - automatic value scaling.
   /// Scales the input values by the ration of : output range/input range
   /// The range is 0,255 for 8u, -128,127 for 8s, ...
   /// The range is 0,1 for float
   void Scale(IImage& Source, IImage& Dest);

   /// From any image type to any image type with given scaling.
   /// Does the conversion Dest = (Source * Ratio) + Offset
   void Scale(IImage& Source, IImage& Dest, int Offset, float Ratio = 0);

   /// Copies an image.
   /// Both images must be of similar types and have the same size
   void Copy(IImage& Source, IImage& Dest);

   /// Copies an image buffer.
   /// Both images must be of the same type, have the same size and the same step
   void Copy(ImageBuffer& Source, ImageBuffer& Dest);

   /// Copies an image to an image buffer.
   /// Both images must be of similar types and have the same size
   /// Dest must not have any padding at the end of the lines, meaning
   /// Dest.Width() must == Dest.ElementStep()
   void Copy(IImage& Source, ImageBuffer& Dest);

   /// Copies an image buffer to an image.
   /// Both images must be of similar types and have the same size
   /// Source must not have any padding at the end of the lines, meaning
   /// Source.Width() must == Source.ElementStep()
   void Copy(ImageBuffer& Source, IImage& Dest);

   /// Converts a color (4 channel) image to a 1 channel image by averaging the first 3 channels
   void ToGray(IImage& Source, IImage& Dest);

   /// Selects 1 channel from a 4 channel image to a 1 channel image - ChannelNo can be from 1 to 4
   void SelectChannel(IImage& Source, IImage& Dest, int ChannelNo);

   /// Converts a 1 channel image to a 4 channel image - first 3 channels of Dest will be set to the value of the first channel of Source
   void ToColor(IImage& Source, IImage& Dest);
};

}
