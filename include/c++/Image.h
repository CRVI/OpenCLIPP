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
/// They can all contain signed or unsigned 8, 16 or 32 bit integer as well as 32 bit float and 64 bit float
/// They can have 1 to 4 channels


#pragma once

#include "Buffer.h"
#include <SImage.h>


namespace OpenCLIPP
{

/// Structure containing an X and Y coordinate
struct CL_API SPoint
{
   uint X, Y;
};

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
   SSize ImageSize() const;   ///< Size of the image, in pixels
   uint Step() const;         ///< Number of bytes between each row
   uint ElementStep() const;  ///< Number of elements between each row
   uint Depth() const;        ///< Number of bits per channel
   uint DepthBytes() const;   ///< Number of bytes per channel
   uint NbChannels() const;   ///< Number of channels in the image, allowed values : 1, 2, 3 or 4
   size_t NbBytes() const;    ///< Total number of bytes for the image
   bool IsFloat() const;      ///< returns true if datatype is floating point
   bool IsUnsigned() const;   ///< returns true if datatype is unsigned integer
   SImage::EDataType DataType() const; ///< Returns a value representing the data type of the image

   operator const SImage& () const;   ///< Returns a SImage that represents this image

protected:
   ImageBase(const SImage& Img);  ///< Constructor - only accessible to derived classes

   SImage m_Img;  ///< The encapsulated structure
};


class CL_API ColorImage;   // Forward declaration

/// Represents an image that is sent to the device as a cl::Buffer
class CL_API Image : public Buffer, public ImageBase
{
public:
   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// The data pointer ImageData is saved for later Send and Read operations
   /// \param CL : A COpenCL instance
   /// \param Img : A SImage representing the host image
   /// \param ImageData : A pointer to where the image data is located
   /// \param flags : Type of OpenCL memory to use, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   Image(COpenCL& CL, const SImage& Img, void * ImageData, cl_mem_flags flags = CL_MEM_READ_WRITE);

protected:
   /// ROI Constructor.
   Image(Image& Img, const SPoint& Offset, const SSize& Size, cl_mem_flags flags);

   /// Constructor for a 3 channel image.
   Image(bool Is3Channel, COpenCL& CL, const SImage& Img, void * ImageData, cl_mem_flags flags);

   friend class ColorImage;   // Some versions of g++ complain that the constructor above is not accessible
};

/// Represents a ROI of an image in the device
class CL_API ImageROI : public Image
{
public:
   /// Constructor.
   /// Creates a ROI of an existing image
   /// The object created with this constructor must not outlive Img
   /// The ROI may be created bigger (start further left) than desired due to memory alignement requirements.
   /// \param CL : A COpenCL instance
   /// \param Img : An Image from which the ROI will be. The ROI object must not outlive this Image object.
   /// \param Offset: Start of the ROI (top-left corner).
   ///         NOTE : Actual start of the ROI may be earlier than asked due to memory alignment requirements.
   /// \param Size: Size of the ROI
   /// \param flags : Type of OpenCL memory to use, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   ImageROI(Image& Img, const SPoint& Offset, const SSize& Size, cl_mem_flags flags = CL_MEM_READ_WRITE);

   /// Read the image from the device memory
   void Read(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   /// Send the image to the device memory
   void Send(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

protected:
   Image& m_Img;
};

/// Represents an in-device only buffer that contains an image
class CL_API TempImage : public Image
{
public:
   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// \param CL : A COpenCL instance
   /// \param Img : A SImage representing the host image
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImage(COpenCL& CL, const SImage& Img, cl_mem_flags flags = CL_MEM_READ_WRITE);

   /// Constructor.
   /// Allocates a buffer in the device memory that can store the image
   /// \param CL : A COpenCL instance
   /// \param Size : Dimention of the image
   /// \param Type : Datatype of the image
   /// \param NbChannels : Desired number of channels
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempImage(COpenCL& CL, SSize Size, SImage::EDataType Type, uint NbChannels = 1, cl_mem_flags flags = CL_MEM_READ_WRITE);

   virtual void SendIfNeeded() { }  ///< Does nothing - it is in-device only and can't be sent

private:
   // Temp images are only inside the device, they can't be sent or read
   void Read(bool = false, std::vector<cl::Event> * = nullptr, cl::Event * = nullptr) { }
   void Send(bool = false, std::vector<cl::Event> * = nullptr, cl::Event * = nullptr) { }
};

/// Represents a 3 channels color image.
/// 3 channel memory acceses are problematic in OpenCL
/// This class serves as 
/// This class will Convert automatically : 
///  - from 3 channels to 4 channels when sending the image to the device
///  - from 4 channels to 3 channels when reading the image from the device
class CL_API ColorImage : public TempImage
{
public:
   /// Constructor.
   /// Allocates, in the device memory, an image to store a 4 channel image
   /// and a buffer to store the 3 channel image.
   /// The data pointer ImageData is saved for later Send and Read operations
   /// \param CL : A COpenCL instance
   /// \param Img : A SImage representing the host image
   /// \param ImageData : A pointer to where the image data is located
   ColorImage(COpenCL& CL, const SImage& Img, void * ImageData);

   /// Read the image from the device memory.
   /// The 4 channel image will be converted to 3 channel and then read to the host.
   /// If blocking is set to true, the Read operation is added to the queue
   /// and then the host waits until the device has finished all outstanding operations.
   /// If blocking is set to false, the Read operation is added to the queue and no wait operation is performed
   /// So if blocking is set to false, the image will not contain the result of the previous kernel execution
   /// \param blocking : Blocking operation
   /// \param events : A list of events that need to be signaled before executing the Read operation
   /// \param event : An event that can be used to wait for the end of the Read operation
   void Read(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   /// Send the image to the device memory.
   /// The image will be sent to a 3 channel buffer in the device and then
   /// converted to a 4 channel image.
   /// Always non-blocking
   /// Events not currently supported when sending ColorImages
   void Send(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

protected:
   Image m_3CImage;   ///< Buffer on the device that contains the 3 channel image
};

}
