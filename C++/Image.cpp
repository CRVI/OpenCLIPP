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
#include "Programs/Color.h"

namespace OpenCLIPP
{

// Helpers
uint DepthOfType(SImage::EDataType Type);    // Gets the number of bits of the given type
SImage TempSImage(SSize Size, SImage::EDataType Type, uint NbChannels = 1);   // Makes a SImage for a temporary (in-device only) image
cl::ImageFormat FormatFromImage(const SImage& Image);
bool IsSupportedFormat(const cl::ImageFormat& inFormat, const cl::Context& context, cl_mem_flags flags);
uint align_step(uint step, uint alignement = 128);


// ImageBase
ImageBase::ImageBase(const SImage& Image)
:  m_Img(Image)
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


// ImageBuffer
ImageBuffer::ImageBuffer(COpenCL& CL, const SImage& Image, void * ImageData, cl_mem_flags flags)
:  Buffer(CL, (char *) ImageData, Image.Height * Image.Step, flags),
   ImageBase(Image)
{ }


// TempImageBuffer
TempImageBuffer::TempImageBuffer(COpenCL& CL, const SImage& Image, cl_mem_flags flags)
:  ImageBuffer(CL, Image, nullptr, flags)
{ }

TempImageBuffer::TempImageBuffer(COpenCL& CL, SSize Size, SImage::EDataType Type,
                                     uint NbChannels, cl_mem_flags flags)
:  ImageBuffer(CL, TempSImage(Size, Type, NbChannels), nullptr, flags)
{ }


// IImage
IImage::IImage(COpenCL& CL, const SImage& Image, cl_mem_flags flags, void * data)
:  ImageBase(Image),
   m_format(FormatFromImage(Image)),
   m_HostBuffer(false),
   m_CL(CL)
{
   Create(flags, data);
}

void IImage::Create(cl_mem_flags flags, void * data)
{
   if (m_format.image_channel_order == CL_RGB)
   {
      // Make it a 4 channel image
      m_format.image_channel_order = CL_RGBA;
      m_Img.Channels = 4;
      m_Img.Step = align_step(Width() * 4 * (Depth() / 8));
   }

   if (!IsSupportedFormat(m_format, m_CL, flags))
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "IImage creation");

   size_t row_pitch = 0;

   if (m_CL.SupportsNoCopy() && data != nullptr)
   {
      // Use HOST_PTR mode to avoid memory transfers
      row_pitch = Step();
      m_HostBuffer = true;
      flags |= CL_MEM_USE_HOST_PTR;
      m_isInDevice = true;
   }
   else
      data = nullptr;

   m_clImage = cl::Image2D(m_CL, flags, m_format, Width(), Height(), row_pitch, data);
}


// TempImage
TempImage::TempImage(COpenCL& CL, const SImage& Image, cl_mem_flags flags)
:  IImage(CL, Image, flags)
{ }

TempImage::TempImage(COpenCL& CL, SSize Size, SImage::EDataType Type, uint NbChannels, cl_mem_flags flags)
:  IImage(CL, TempSImage(Size, Type, NbChannels), flags)
{ }


// Image
Image::Image(COpenCL& CL, const SImage& Image, void * ImageData, cl_mem_flags flags)
:  IImage(CL, Image, flags, ImageData),
   m_data(ImageData)
{
   if (Image.Channels == 3)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image can't be 3 channels - use ColorImage to handle 3 channel images");
}

// Read the image from the device memory
void Image::Read(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   if (m_data == nullptr)
      return;

   cl::size_t<3> origin, region;
   region[0] = Width();
   region[1] = Height();
   region[2] = 1;

   if (!m_HostBuffer)
      m_CL.GetQueue().enqueueReadImage(m_clImage, (cl_bool) blocking, origin, region, Step(), 0, m_data, events, event);
   else
   {
      if (m_CL.SupportsNoCopy())
      {
         // The device uses the same memory for the device and the host, no transfer is needed
         if (blocking)
            m_CL.GetQueue().finish();

         return;
      }

      cl::Event unmapEvent;
      std::vector<cl::Event> mapEvent(1, cl::Event());
      size_t row_pitch = 0;
      m_CL.GetQueue().enqueueMapImage(m_clImage, (cl_bool) blocking, CL_MAP_READ, origin, region, &row_pitch, 0, events, &mapEvent[0]);

      if (row_pitch != Step())
         throw cl::Error(CL_MISALIGNED_SUB_BUFFER_OFFSET, "Row pitch does not match during image reading");

      m_CL.GetQueue().enqueueUnmapMemObject(m_clImage, m_data, &mapEvent, &unmapEvent);

      if (blocking)
         unmapEvent.wait();

      if (event != nullptr)
         *event = unmapEvent;
   }

}

// Send the image to the device memory
void Image::Send(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   if (m_data == nullptr)
      return;

   cl::size_t<3> origin, region;
   region[0] = Width();
   region[1] = Height();
   region[2] = 1;

   if (!m_HostBuffer)
      m_CL.GetQueue().enqueueWriteImage(m_clImage, (cl_bool) blocking, origin, region, Step(), 0, m_data, events, event);
   else
   {
      if (m_CL.SupportsNoCopy())
      {
         // The device uses the same memory for the device and the host, no transfer is needed
         if (blocking)
            m_CL.GetQueue().finish();

         m_isInDevice = true;
         return;
      }

      cl::Event unmapEvent;
      std::vector<cl::Event> mapEvent(1, cl::Event());
      size_t row_pitch = 0;
      m_CL.GetQueue().enqueueMapImage(m_clImage, (cl_bool) blocking, CL_MAP_WRITE, origin, region, &row_pitch, 0, events, &mapEvent[0]);

      if (row_pitch != Step())
         throw cl::Error(CL_MISALIGNED_SUB_BUFFER_OFFSET, "Row pitch does not match during image sending");

      m_CL.GetQueue().enqueueUnmapMemObject(m_clImage, m_data, &mapEvent, &unmapEvent);

      if (blocking)
         unmapEvent.wait();

      if (event != nullptr)
         *event = unmapEvent;
   }

   m_isInDevice = true;
}

void Image::SendIfNeeded()
{
   if (!m_isInDevice)
      Send();
}



// ColorImage
ColorImage::ColorImage(COpenCL& CL, const SImage& Image, void * ImageData)
:  TempImage(CL, Image, CL_MEM_READ_WRITE),
   m_Buffer(CL, Image, ImageData, CL_MEM_READ_WRITE)
{
   if (Image.Channels != 3)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "ColorImage received an image with a number of channels different than 3");
}

// Read the image from the device memory
void ColorImage::Read(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   // NOTE : Synchronisation is not good here because conversion will start without waiting on the events
   m_CL.GetColorConverter().Convert4CTo3C(*this, m_Buffer);
   m_Buffer.Read(blocking, events, event);

   m_isInDevice = true;
}

// Send the image to the device memory
void ColorImage::Send()
{
   m_Buffer.Send();
   m_CL.GetColorConverter().Convert3CTo4C(m_Buffer, *this);
}

void ColorImage::SendIfNeeded()
{
   if (!m_isInDevice)
      Send();
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
   case SImage::NbDataTypes:
   default:
      return 0;
   }
}

SImage TempSImage(SSize Size, SImage::EDataType Type, uint NbChannels)
{
   SImage Image = {Size.Width, Size.Height, DepthOfType(Type) * NbChannels * Size.Width / 8, NbChannels, Type};
   return Image;
}

inline bool operator == (const cl::ImageFormat& left, const cl::ImageFormat& right)
{
   return (left.image_channel_data_type == right.image_channel_data_type &&
      left.image_channel_order == right.image_channel_order);
}

bool IsSupportedFormat(const cl::ImageFormat& inFormat, const cl::Context& context, cl_mem_flags flags)
{
   std::vector<cl::ImageFormat> formats;
   context.getSupportedImageFormats(flags, CL_MEM_OBJECT_IMAGE2D, &formats);

   for (auto format : formats)
      if (inFormat == format)
         return true;

   return false;
}

cl::ImageFormat FormatFromImage(const SImage& image)
{
   cl::ImageFormat format;

   switch (image.Type)
   {
   case SImage::U8:
      format.image_channel_data_type = CL_UNSIGNED_INT8;
      break;
   case SImage::S8:
      format.image_channel_data_type = CL_SIGNED_INT8;
      break;
   case SImage::U16:
      format.image_channel_data_type = CL_UNSIGNED_INT16;
      break;
   case SImage::S16:
      format.image_channel_data_type = CL_SIGNED_INT16;
      break;
   case SImage::U32:
      format.image_channel_data_type = CL_UNSIGNED_INT32;
      break;
   case SImage::S32:
      format.image_channel_data_type = CL_SIGNED_INT32;
      break;
   case SImage::F32:
      format.image_channel_data_type = CL_FLOAT;
      break;
   case SImage::NbDataTypes:
   default:
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "FormatFromILImage - DataType");
   }

   switch (image.Channels)
   {
   case 1:
      format.image_channel_order = CL_R;
      break;
   case 2:
      format.image_channel_order = CL_RA;
   case 3:
      // NOTE : 3 channel images are not supported on the devices - they need to be converted to 4 channels
      // we allow it here to mark this image as a 3 channel buffer
      format.image_channel_order = CL_RGB;
      break;
   case 4:
      format.image_channel_order = CL_RGBA;
      break;
   default:
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "FormatFromILImage - Channels");
   }

   return format;
}

uint align_step(uint step, uint alignement)
{
   uint mod = step % alignement;
   if (mod == 0)
      return step;

   return step + alignement - mod;
}

}
