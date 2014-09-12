////////////////////////////////////////////////////////////////////////////////
//! @file	: Buffer.h 
//! @date   : Jul 2013
//!
//! @brief  : Objects that represent a memory buffer in the OpenCL device
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

#include "OpenCL.h"

namespace OpenCLIPP
{

/// Base class for objects representing a memory space in the device, like an Image or a Buffer.
/// Also keeps track of presence of valid data in the device
/// Not useable directly
class CL_API Memory
{
public:
   virtual ~Memory() { }   ///< Destructor

   /// Returns In device memory state.
   /// True when the memory in the device contains meaningful data - after an data has been sent to the device or contains
   bool IsInDevice() const;

   /// Sets In device memory state.
   /// Sets the value of InDevice, to be used after a kernel has filled it with data
   void SetInDevice(bool inDevice = true);

   /// Sends to the device if needed.
   /// Sends the data to the device if IsInDevice() is false - only useful for objects that have a Send() method
   virtual void SendIfNeeded() { }

protected:
   Memory();   ///< Constructor - useable by derived classes only

   /// Tracks In device memory state.
   /// True when the memory in the device contains meaningful data
   /// (ie: true after an image has been sent to the device, false for an unitialised temporary image)
   bool m_isInDevice;   
};

/// Base class for buffer objects - Wraps a cl::Buffer
class CL_API IBuffer : public Memory
{
public:
   /// Converts to a cl::Buffer
   operator cl::Buffer& ()
   {
      return m_Buffer;
   }

   /// Converts to a cl_mem
   operator cl_mem ()
   {
      return (cl_mem) m_Buffer();
   }

   /// Returns the size in bytes
   size_t Size() const
   {
      return m_Size;
   }
   
protected:
   /// Constructor - useable by derived classes only
   /// \param CL : A COpenCL instance
   /// \param size : Size of the buffer in bytes
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   /// \param data : Pointer to the array in host memory
   /// \param copy : If we want a copy of the data currently on the host (will use CL_MEM_COPY_HOST_PTR)
   IBuffer(COpenCL& CL, size_t size, cl_mem_flags flags, void * data = nullptr, bool copy = false);

   /// Constructor for creating a SubBuffer - useable by derived classes only
   /// \param CL : A COpenCL instance
   /// \param MainBuffer : Buffer from which the SubBuffer will be created
   /// \param offset : Offset in bytes from the start of the main buffer
   /// \param size : Size of the buffer in bytes
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   IBuffer(COpenCL& CL, IBuffer& MainBuffer, size_t& offset, size_t& size, cl_mem_flags flags);

   cl::Buffer m_Buffer;    ///< The encapsulated OpenCL buffer object
   size_t m_Size;          ///< The size of the buffer, in bytes
   bool m_HostBuffer;      ///< true when using CL_MEM_USE_HOST_PTR and mapped memory for faster memory transfer
};

/// Represents an in-device only memory buffer
class CL_API TempBuffer : public IBuffer
{
public:
   /// Constructor.
   /// \param CL : A COpenCL instance
   /// \param size : Size of the buffer in bytes
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   TempBuffer(COpenCL& CL, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
};

/// Represents a buffer that can be sent to the device and read from the device
class CL_API Buffer : public IBuffer
{
public:
   /// Constructor.
   /// \param CL : A COpenCL instance
   /// \param data : Pointer to the array in host memory
   /// \param length : Length of the array in elements (not in bytes)
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   template<class T>
   Buffer(COpenCL& CL, T * data, size_t length, cl_mem_flags flags = CL_MEM_READ_WRITE);
   
   /// Read the buffer from the device memory
   virtual void Read(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   /// Send the buffer to the device memory
   virtual void Send(bool blocking = false, std::vector<cl::Event> * events = nullptr, cl::Event * event = nullptr);

   virtual void SendIfNeeded();  ///< Sends the data to the device if IsInDevice() is false

protected:
   /// Constructor for creating a SubBuffer - useable by derived classes only
   /// \param CL : A COpenCL instance
   /// \param MainBuffer : Buffer from which the SubBuffer will be created
   /// \param offset : Offset in bytes from the start of the main buffer
   /// \param size : Size of the buffer in bytes
   /// \param flags : Type of OpenCL memory to create, allowed values : CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY
   /// \param data : Pointer to the array in host memory
   /// \param copy : If we want a copy of the data currently on the host (will use CL_MEM_COPY_HOST_PTR)
   Buffer(COpenCL& CL, Buffer& MainBuffer, size_t offset, size_t size, cl_mem_flags flags, size_t& outNewOffset);

   COpenCL& m_CL;    ///< The COpenCL instance this image is assotiated to
   void * m_data;    ///< Pointer to the buffer data on the host

   void operator = (const Buffer&) { }  ///< Not a copyable object
};

/// Represents a buffer that is automatically sent to the device but can't be read back - is read-only from the device
class CL_API ReadBuffer : public IBuffer
{
public:
   /// Constructor.
   /// Creates a buffer on the device to contain the data
   /// Copies the buffer on the host side
   /// Issues a Send() from the copy to the device
   /// After the object is created, the array pointed to by data can be freed
   /// \param CL : A COpenCL instance
   /// \param data : Pointer to the array in host memory
   /// \param length : Length of the array in elements (not in bytes)
   template<class T>
   ReadBuffer(COpenCL& CL, T * data, size_t length);
};

template<class T>
inline Buffer::Buffer(COpenCL& CL, T * data, size_t length, cl_mem_flags flags)
:  IBuffer(CL, sizeof(T) * length, flags, data),
   m_CL(CL),
   m_data(data)
{ }

template<class T>
inline ReadBuffer::ReadBuffer(COpenCL& CL, T * data, size_t length)
:  IBuffer(CL, sizeof(T) * length, CL_MEM_READ_ONLY, data, true)
{ }

}
