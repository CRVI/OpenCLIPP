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
#include "Programs/Conversions.h"

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

SSize ImageBase::ImageSize() const
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
{
   if (Img.Channels < 1 || Img.Channels > 4)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Invalid number of channels");

   if (Img.Channels == 3)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Please use ColorImage for 3 channel images");

   if (Img.Type < 0 || Img.Type >= Img.NbDataTypes)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Invalid data type");
}

Image::Image(bool Is3Channel, COpenCL& CL, const SImage& Img, void * ImageData, cl_mem_flags flags)
:  Buffer(CL, (char *) ImageData, Img.Height * Img.Step, flags),
   ImageBase(Img)
{
   assert(Is3Channel);

   if (Img.Type < 0 || Img.Type >= Img.NbDataTypes)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Invalid data type");
}

// ROI Constructor
Image::Image(Image& Img, const SPoint& Offset, const SSize& Size, cl_mem_flags flags)
:  Buffer(Img.m_CL, Img,
          Offset.Y * Img.Step() + Offset.X * Img.DepthBytes() * Img.NbChannels(),   // Calculate offset in bytes
          Size.Height * Img.Step() - (Img.Width() - Size.Width) * Img.DepthBytes() * Img.NbChannels(),   // Calculate size in bytes
          flags,
          (size_t&) flags),    // Use flags as temporary variable to receive the actual offset value
   ImageBase(Img)
{
   // Buffer is not always capable of creating a sub-buffer at the offset specified due to memory alignement requirements
   // So it may create a buffer that starts ealier than the offset we calculated and thus it will have a bigger size
   // So in order to cover the whole ROI, we need to recalculate the size of the image
   size_t AskedOffset = Offset.Y * Img.Step() + Offset.X * Img.DepthBytes() * Img.NbChannels();
   size_t ActualBufferOffset = size_t(flags);
   size_t OffsetDifference = AskedOffset - ActualBufferOffset;
   size_t OffsetDiffElements = OffsetDifference / (Img.DepthBytes() * Img.NbChannels());

   if (OffsetDiffElements <= Offset.X)
   {
      m_Img.Width = Size.Width + uint(OffsetDiffElements);
      m_Img.Height = Size.Height;
      return;
   }

   // Offset difference covers more than one row - this would be difficult to handle
   throw cl::Error(CL_MISALIGNED_SUB_BUFFER_OFFSET, "Unable to reate ROI due to memory alignement requirements");
}

// ImageROI
ImageROI::ImageROI(Image& Img, const SPoint& Offset, const SSize& Size, cl_mem_flags flags)
:  Image(Img, Offset, Size, flags),
   m_Img(Img)
{ }

/// Read the image from the device memory
void ImageROI::Read(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   m_Img.Read(blocking, events, event);
}

/// Send the image to the device memory
void ImageROI::Send(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   m_Img.Send(blocking, events, event);
}


// TempImage
TempImage::TempImage(COpenCL& CL, const SImage& Img, cl_mem_flags flags)
:  Image(CL, Img, nullptr, flags)
{ }

TempImage::TempImage(COpenCL& CL, SSize Size, SImage::EDataType Type, uint NbChannels, cl_mem_flags flags)
:  Image(CL, TempSImage(Size, Type, NbChannels), nullptr, flags)
{ }


// ColorImage
SSize SizeOfImage(const SImage& Img)
{
   SSize size = {Img.Width, Img.Height};
   return size;
}

ColorImage::ColorImage(COpenCL& CL, const SImage& Img, void * ImageData)
:  TempImage(CL, SizeOfImage(Img), Img.Type, 4, CL_MEM_READ_WRITE),
   m_3CImage(true, CL, Img, ImageData, CL_MEM_READ_WRITE)
{
   if (Img.Channels != 3)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "ColorImage received an image with a number of channels different than 3");
}

// Read the image from the device memory
void ColorImage::Read(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   // NOTE : Synchronisation is not good here because conversion will start without waiting on the events
   m_CL.GetConverter().Copy4Cto3C(*this, m_3CImage);
   m_3CImage.Read(blocking, events, event);

   m_isInDevice = true;
}

// Send the image to the device memory
void ColorImage::Send(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   // Sending ColorImages is currently always non-blocking and does not support events
   assert(!blocking);
   assert(events == nullptr);
   assert(event == nullptr);

   m_3CImage.Send();
   m_CL.GetConverter().Copy3Cto4C(m_3CImage, *this);
}


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
