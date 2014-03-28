////////////////////////////////////////////////////////////////////////////////
//! @file	: SImage.h 
//! @date   : Jul 2013
//!
//! @brief  : Declaration of structure SImage, used to describe an image
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

#ifndef __SIMAGE__H
#define __SIMAGE__H

struct SImage
{
   uint Width;    ///< Width of the image, in pixels
   uint Height;   ///< Height of the image, in pixels
   uint Step;     ///< Nb of bytes between each row
   uint Channels; ///< Number of channels in the image, allowed values : 1, 2, 3 or 4

   /// EDataType : Lists possible types of data
   enum EDataType
   {
      U8,            /// Unsigned 8-bit integer (unsigned char)
      S8,            /// Signed 8-bit integer (char)
      U16,           /// Unsigned 16-bit integer (unsigned short)
      S16,           /// Signed 16-bit integer (short)
      U32,           /// Unsigned 32-bit integer (unsigned int)
      S32,           /// Signed 32-bit integer (int)
      F32,           /// 32-bit floating point (float)
      F64,           /// 64-bit floating point (double)
      NbDataTypes,   /// Number of possible data types
   } Type;  ///< Data type of each channel in the image
};

#endif   // __SIMAGE__H
