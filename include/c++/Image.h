////////////////////////////////////////////////////////////////////////////////
//! @file	: Image.h 
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

/// These classes are used to represent images that are in the computing device
/// They allow sending images to the device and reading images from the device
/// Some classes represent temporary objects that reside only in the device memory
/// They can all contain signed or unsigned 8, 16 or 32 bit integer as well as 32 bit float
/// They can have 1 or 4 channels
/// 3 channel images can be represented by ColorImage

/// Differences between a Buffer, an Image and an ImageBuffer :
///   A Buffer is a memory region on the device, it is used to store an array.
///   The device does not 'know' the type of data inside the buffer, it know only the number of bytes it has.
///   -
///   An ImageBuffer is a buffer that contains image data
///   The device sees the ImageBuffer as a simple Buffer :
///      It does not know what is in the buffer
///   Operations on ImageBuffers can be faster than operations on Images because the kernels can do vector processing
///   -
///   An Image uses the 'texture' hardware of the device (if available).
///   The device knows the pixel type, the step between lines and other information about the image.
///   Some operations are faster on images because they benefit from the intelligent cache for texture that is present on GPUs.
///   Kernel implementation of image processing is simpler on Images


#pragma once

#include "Buffer.h"
#include <SImage.h>


namespace OpenCLIPP
{

/// Structure containing the size of an image - in pixels
struct CL_API SSize
{
   uint Width;    ///< Width - in pixels
   uint Height;   ///< Height - in pixels
};


/// Base class for all images - encapsulates a SImage
class CL_API ImageBase
{
public:
   cl::NDRange FullRange() const;         ///< Returns a 2D global range for 1 worker per pixel
   cl::NDRange VectorRange(int NbElementsPerWorker) const;  ///< Returns a 2D global range for multiple pixels per worker

   uint Width() const;        ///< Width of the image, in pixels
   uint Height() const;       ///< Height of the image, in pixels
   SSize Size() const;        ///< Size of the image, in pixels
   uint Step() const;         ///< Number of bytes between each row
   uint ElementStep() const;  ///< Number of elemets between each row
   uint Depth() const;        ///< Number of bits per channel
   uint DepthBytes() const;   ///< Number of bytes per channel
   uint NbChannels() const;   ///< Number of channels in the image, allowed values : 1, 2, 3 or 4
   size_t NbBytes() const;    ///< Total number of bytes for the image
   bool IsFloat() const;      ///< returns true if datatype is floating point
   bool IsUnsigned() const;   ///< returns true if datatype is unsigned integer
   SImage::EDataType DataType() const; ///< Returns a value representing the data type of the image

   operator const SImage& () const;   ///< Returns a SImage that represents this image

protected:
   ImageBase(const SImage& Image);  ///< Constructor - only accessible to derived classes

   SImage m_Img;  ///< The encapsulated structure
};



/// Represents an image that is sent to the device as a cl::Buffer
class CL_API ImageBuffer : public Buffer, public ImageBase
{
public:
   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// The data pointer in image is saved for later Send and Read operations
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param ImageData : A pointer to where the image data is located
   /// \param flags : Type of OpenCL memory to use, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   ImageBuffer(COpenCL& CL, const SImage& Image, void * ImageData, cl_mem_flags flags = CL_MEM_READ_WRITE);
};


/// Represents an in-device only buffer that contains an image
class CL_API TempImageBuffer : public ImageBuffer
{
public:
   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImageBuffer(COpenCL& CL, const SImage& Image, cl_mem_flags flags = CL_MEM_READ_WRITE);

   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// \param CL : A COpenCL instance
   /// \param Size : Dimention of the image
   /// \param Type : Datatype of the image
   /// \param NbChannels : Desired number of channels
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImageBuffer(COpenCL& CL, SSize Size, SImage::EDataType Type, uint NbChannels = 1, cl_mem_flags flags = CL_MEM_READ_WRITE);

   virtual void SendIfNeeded() { }  ///< Does nothing - it is in-device only and can't be sent

private:
   // This buffer can't be sent or read from the device
   void Read(bool = false, std::vector<cl::Event> * = nullptr, cl::Event * = nullptr) { }
   void Send(bool = false, std::vector<cl::Event> * = nullptr, cl::Event * = nullptr) { }
};

}
