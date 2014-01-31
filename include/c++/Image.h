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
   cl::NDRange FullRange();         ///< Returns a 2D global range for 1 worker per pixel
   cl::NDRange VectorRange(int NbElementsPerWorker);  ///< Returns a 2D global range for multiple pixels per worker

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


/// Base class for Images (not including image buffers) - Wraps cl::Image2D - not useable directly
class CL_API IImage : public ImageBase, public Memory
{
public:
   /// Converts to a cl::Image2D
   operator cl::Image2D& ()   
   {
      return m_clImage;
   }

   /// Converts to a cl_mem
   operator cl_mem () 
   {
      return (cl_mem) m_clImage();
   }

protected:

   /// Constructor.
   /// Allocates an image in the device memory
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   /// \param data : A pointer to where the image data is located
   IImage(COpenCL& CL, const SImage& Image, cl_mem_flags flags = CL_MEM_READ_WRITE, void * data = nullptr);

   /// Allocates the image in the device memory - called automatically by the contructor
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   /// \param data : A pointer to where the image data is located
   void Create(cl_mem_flags flags, void * data = nullptr);

   cl::ImageFormat m_format;  ///< Format of the image
   cl::Image2D m_clImage;     ///< The encapsulated OpenCL image object
   bool m_HostBuffer;         ///< true when using CL_MEM_USE_HOST_PTR and mapped image for faster image transfer
   COpenCL& m_CL;             ///< The COpenCL instance this image is assotiated to

   void operator = (const IImage&) { }  ///< Not a copyable object
};

/// A temporary image.
/// Represents a temporary image in the device that is the same size and type as the given SImage
/// Can't be transferred back to host memory
/// If the given image is a 3C color image, this object represents a 4C color image instead
class CL_API TempImage : public IImage
{
public:
   /// Constructor.
   /// Allocates an image in the device memory
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImage(COpenCL& CL, const SImage& Image, cl_mem_flags flags = CL_MEM_READ_WRITE);

   /// Constructor.
   /// Allocates an image in the device memory
   /// \param CL : A COpenCL instance
   /// \param Size : Dimention of the image
   /// \param Type : Datatype of the image
   /// \param NbChannels : Desired number of channels
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImage(COpenCL& CL, SSize Size, SImage::EDataType Type, uint NbChannels = 1, cl_mem_flags flags = CL_MEM_READ_WRITE);
};


/// Represents an image in the device as a cl::Image2D (as a texture)
class CL_API Image : public IImage
{
public:
   /// Constructor.
   /// Allocates an image in the device memory
   /// The data pointer in image is saved for later Send and Read operations
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param ImageData : A pointer to where the image data is located
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   Image(COpenCL& CL, const SImage& Image, void * ImageData, cl_mem_flags flags = CL_MEM_READ_WRITE);

   /// Read the image from the device memory.
   /// If blocking is set to true, the Read operation is added to the queue
   /// and then the host waits until the device has finished all outstanding operations.
   /// If blocking is set to false, the Read operation is added to the queue and no wait operation is performed
   /// So if blocking is set to false, the image will not contain the result of the previous kernel execution
   /// \param blocking : Blocking operation
   /// \param events : A list of events that need to be signaled before executing the Read operation
   /// \param event : An event that can be used to wait for the end of the Read operation
   void Read(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   /// Send the image to the device memory.
   /// If blocking is set to true, the Send operation is added to the queue
   /// and then the host waits until the device has finished all outstanding operations.
   /// If blocking is set to false, the Send operation is added to the queue and no wait operation is performed
   /// blocking normally does not need to be set to true for the data to be available as input of later kernel execution
   /// because OpenCL executes all operations in-order except if the queue has been created Out-of-Order
   /// (which is currently not supported by this library).
   /// \param blocking : Blocking operation
   /// \param events : A list of events that need to be signaled before executing the Read operation
   /// \param event : An event that can be used to wait for the end of the Read operation
   void Send(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   virtual void SendIfNeeded();  ///< Sends the data to the device if IsInDevice() is false

protected:
   void * m_data;    ///< Pointer to the image data on the host
};


/// Represents a 3 channels color image.
/// OpenCL supports only 1, 2 or 4 channel images
/// This class will Convert automatically : 
///  - from 3 channels to 4 channels when sending the image to the device
///  - from 4 channels to 3 channels when reading the image from the device
class CL_API ColorImage : public TempImage
{
public:
   /// Constructor.
   /// Allocates, in the device memory, an image to store a 4 channel image
   /// and an image buffer to store the 3 channel image.
   /// The data pointer in image is saved for later Send and Read operations
   /// \param CL : A COpenCL instance
   /// \param Image : A SImage representing the host image
   /// \param ImageData : A pointer to where the image data is located
   ColorImage(COpenCL& CL, const SImage& Image, void * ImageData);

   /// Read the image from the device memory.
   /// The 4 channel image will be converted to the 3 channel image buffer and then read to the host.
   /// If blocking is set to true, the Read operation is added to the queue
   /// and then the host waits until the device has finished all outstanding operations.
   /// If blocking is set to false, the Read operation is added to the queue and no wait operation is performed
   /// So if blocking is set to false, the image will not contain the result of the previous kernel execution
   /// \param blocking : Blocking operation
   /// \param events : A list of events that need to be signaled before executing the Read operation
   /// \param event : An event that can be used to wait for the end of the Read operation
   void Read(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   /// Send the image to the device memory.
   /// The image will be sent to a 3 channel image buffer in the device and then
   /// converted to a 4 channel image.
   /// Always non-blocking
   void Send();

   virtual void SendIfNeeded();  ///< Sends the data to the device if IsInDevice() is false

protected:
   ImageBuffer m_Buffer;   ///< Buffer on the device that contains the 3 channel image

private:
   void Send(bool, std::vector<cl::Event> * = nullptr, cl::Event * = nullptr);    // Only non-blocking non-synced send available for color images right now
};

}
