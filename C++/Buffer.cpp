////////////////////////////////////////////////////////////////////////////////
//! @file	: Buffer.cpp
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

#include "Buffer.h"

namespace OpenCLIPP
{

// Memory
Memory::Memory()
:  m_isInDevice(false)
{ }

bool Memory::IsInDevice() const
{
   return m_isInDevice;
}

void Memory::SetInDevice(bool inDevice)
{
   m_isInDevice = inDevice;
}


// IBuffer
IBuffer::IBuffer(COpenCL& CL, size_t size, cl_mem_flags flags, void * data, bool copy)
:  m_Size(size),
   m_HostBuffer(false)
{
   if (copy)
   {
      // OpenCL will make a copy of the data and will automatically transfer the data to the device when needed
      m_Buffer = cl::Buffer(CL, flags | CL_MEM_COPY_HOST_PTR, size, data);
      m_isInDevice = true;
   }
   else
   {
      if (CL.SupportsNoCopy() && data != nullptr)
      {
         // Use HOST_PTR mode to avoid memory transfers
         flags |= CL_MEM_USE_HOST_PTR;
         m_HostBuffer = true;
         m_isInDevice = true;
      }
      else
         data = nullptr;

      m_Buffer = cl::Buffer(CL, flags, size, data);
   }

}

// ROI constructor
IBuffer::IBuffer(COpenCL& CL, IBuffer& MainBuffer, size_t& offset, size_t& size, cl_mem_flags flags)
:  m_Size(size),
   m_HostBuffer(false)
{

   {
      // createSubBuffer accepts only origin values that align with the number of bits specified in value CL_DEVICE_MEM_BASE_ADDR_ALIGN
      // So we adjust the origin of the buffer to fit the memory alignment

      // Retrieve number of bits
      const cl::Device& device = CL;
      uint AlignBits = device.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();

      // Calculate difference
      size_t mod = offset % (AlignBits / 8);

      // Adjust origin and size (ROI will start earlier than what was demanded)
      offset -= mod;
      size += mod;
   }

   struct {
      size_t origin;
      size_t size;
   } cl_buffer_region = {offset, size};

   m_Buffer = MainBuffer.m_Buffer.createSubBuffer(flags, CL_BUFFER_CREATE_TYPE_REGION, &cl_buffer_region);
}


// TempBuffer
TempBuffer::TempBuffer(COpenCL& CL, size_t size, cl_mem_flags flags)
:  IBuffer(CL, size, flags)
{ }


// Buffer

// ROI constructor
Buffer::Buffer(COpenCL& CL, Buffer& MainBuffer, size_t offset, size_t size, cl_mem_flags flags, size_t& outNewOffset)
:  IBuffer(CL, MainBuffer, offset, size, flags),
   m_CL(CL),
   m_data(((char*)MainBuffer.m_data) + (MainBuffer.m_data ? offset : 0))   // Add offset only if not null
{
   outNewOffset = offset;
}

// Read the buffer from the device memory
void Buffer::Read(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   if (m_data == nullptr)
      return;

   if (!m_HostBuffer)
      m_CL.GetQueue().enqueueReadBuffer(m_Buffer, (cl_bool) blocking, 0, m_Size, m_data, events, event);
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
      m_CL.GetQueue().enqueueMapBuffer(m_Buffer, (cl_bool) blocking, CL_MAP_READ, 0, m_Size, events, &mapEvent[0]);
      m_CL.GetQueue().enqueueUnmapMemObject(m_Buffer, m_data, &mapEvent, &unmapEvent);

      if (blocking)
         unmapEvent.wait();

      if (event != nullptr)
         *event = unmapEvent;
   }

}

// Send the buffer to the device memory
void Buffer::Send(bool blocking, std::vector<cl::Event> * events, cl::Event * event)
{
   if (m_data == nullptr)
      return;

   if (!m_HostBuffer)
      m_CL.GetQueue().enqueueWriteBuffer(m_Buffer, (cl_bool) blocking, 0, m_Size, m_data, events, event);
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
      m_CL.GetQueue().enqueueMapBuffer(m_Buffer, (cl_bool) blocking, CL_MAP_WRITE, 0, m_Size, events, &mapEvent[0]);
      m_CL.GetQueue().enqueueUnmapMemObject(m_Buffer, m_data, &mapEvent, &unmapEvent);

      if (blocking)
         unmapEvent.wait();

      if (event != nullptr)
         *event = unmapEvent;
   }

   m_isInDevice = true;
}

void Buffer::SendIfNeeded()
{
   if (!m_isInDevice)
      Send();
}

}
