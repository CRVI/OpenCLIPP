////////////////////////////////////////////////////////////////////////////////
//! @file	: ConversionsBuffer.h
//! @date   : Jun 2014
//!
//! @brief  : Image depth conversion
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

#pragma once

#include "Program.h"

namespace OpenCLIPP
{

/// A program used to Convert the type of an image
class CL_API ConversionsBuffer : public ImageBufferProgram
{
public:
   ConversionsBuffer(COpenCL& CL)
   :  ImageBufferProgram(CL, "Convert_Buffer.cl")
   { }

   /// From any image type to any image type - no value scaling
   void Convert(ImageBuffer& Source, ImageBuffer& Dest);

   /// From any image type to any image type - automatic value scaling.
   /// Scales the input values by the ration of : output range/input range
   /// The range is 0,255 for 8u, -128,127 for 8s, ...
   /// The range is 0,1 for float
   void Scale(ImageBuffer& Source, ImageBuffer& Dest);

   /// From any image type to any image type with given scaling.
   /// Does the conversion Dest = (Source * Ratio) + Offset
   void Scale(ImageBuffer& Source, ImageBuffer& Dest, int Offset, float Ratio = 0);

   /// Copies an image.
   /// Both images must be of the same type and have the same size
   void Copy(ImageBuffer& Source, ImageBuffer& Dest);

   /// Converts a color (4 channel) image to a 1 channel image by averaging the first 3 channels
   void ToGray(ImageBuffer& Source, ImageBuffer& Dest);

   /// Selects 1 channel from a 4 channel image to a 1 channel image - ChannelNo can be from 1 to 4
   void SelectChannel(ImageBuffer& Source, ImageBuffer& Dest, int ChannelNo);

   /// Converts a 1 channel image to a 4 channel image - first 3 channels of Dest will be set to the value of the first channel of Source
   void ToColor(ImageBuffer& Source, ImageBuffer& Dest);
};

}
