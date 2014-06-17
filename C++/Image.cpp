////////////////////////////////////////////////////////////////////////////////
//! @file	: Image.cpp
//! @date   : Jul 2013
//!
//! @brief  : Objects that represent an image in the OpenCL device
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

#include "Image.h"

namespace OpenCLIPP
{

// Helpers
uint DepthOfType(SImage::EDataType Type);    // Gets the number of bits of the given type
SImage TempSImage(SSize Size, SImage::EDataType Type, uint NbChannels = 1);   // Makes a SImage for a temporary (in-device only) image


// ImageBase
ImageBase::ImageBase(const SImage& Img)
:  m_Img(Img)
{ }

cl::NDRange ImageBase::FullRange() const
{
   return cl::NDRange(Width(), Height(), 1);
}

cl::NDRange ImageBase::VectorRange(int NbElementsPerWorker) const
{
   return cl::NDRange(Width() * NbChannels() / NbElementsPerWorker, Height(), 1);
}

uint ImageBase::Width() const
{
   return m_Img.Width;
}

uint ImageBase::Height() const
{
   return m_Img.Height;
}

SSize ImageBase::Size() const
{
   SSize Size = {m_Img.Width, m_Img.Height};
   return Size;
}

uint ImageBase::Step() const
{
   return m_Img.Step;
}

uint ImageBase::ElementStep() const
{
   return Step() / DepthBytes();
}

uint ImageBase::Depth() const
{
   return DepthOfType(m_Img.Type);
}

uint ImageBase::DepthBytes() const
{
   assert(Depth() % 8 == 0);
   return Depth() / 8;
}

uint ImageBase::NbChannels() const
{
   return m_Img.Channels;
}

size_t ImageBase::NbBytes() const
{
   return size_t(Height()) * Step();
}

bool ImageBase::IsFloat() const
{
   switch (m_Img.Type)
   {
   case SImage::U8:
   case SImage::U16:
   case SImage::U32:
   case SImage::S8:
   case SImage::S16:
   case SImage::S32:
      return false;
   case SImage::F32:
   case SImage::F64:
      return true;
   case SImage::NbDataTypes:
   default:
      return false;
   }

}

bool ImageBase::IsUnsigned() const
{
   switch (m_Img.Type)
   {
   case SImage::U8:
   case SImage::U16:
   case SImage::U32:
      return true;

   case SImage::S8:
   case SImage::S16:
   case SImage::S32:
   case SImage::F32:
   case SImage::F64:
      return false;
   case SImage::NbDataTypes:
   default:
      return false;
   }

}

SImage::EDataType ImageBase::DataType() const
{
   return m_Img.Type;
}

ImageBase::operator const SImage& () const
{
   return m_Img;
}


// Image
Image::Image(COpenCL& CL, const SImage& Img, void * ImageData, cl_mem_flags flags)
:  Buffer(CL, (char *) ImageData, Img.Height * Img.Step, flags),
   ImageBase(Img)
{ }


// TempImage
TempImage::TempImage(COpenCL& CL, const SImage& Img, cl_mem_flags flags)
:  Image(CL, Img, nullptr, flags)
{ }

TempImage::TempImage(COpenCL& CL, SSize Size, SImage::EDataType Type,
                                     uint NbChannels, cl_mem_flags flags)
:  Image(CL, TempSImage(Size, Type, NbChannels), nullptr, flags)
{ }



// Helpers
uint DepthOfType(SImage::EDataType Type)
{
   switch (Type)
   {
   case SImage::U8:
   case SImage::S8:
      return 8;
   case SImage::U16:
   case SImage::S16:
      return 16;
   case SImage::U32:
   case SImage::S32:
   case SImage::F32:
      return 32;
   case SImage::F64:
      return 64;
   case SImage::NbDataTypes:
   default:
      return 0;
   }
}

SImage TempSImage(SSize Size, SImage::EDataType Type, uint NbChannels)
{
   SImage Img = {Size.Width, Size.Height, DepthOfType(Type) * NbChannels * Size.Width / 8, NbChannels, Type};
   return Img;
}

}
