////////////////////////////////////////////////////////////////////////////////
//! @file	: OpenCL.cpp
//! @date   : Jul 2013
//!
//! @brief  : COpenCL object - takes care of initializing OpenCL
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

#include "OpenCL.h"
#include "Programs/Conversions.h"

#include <string>


using namespace std;


namespace OpenCLIPP
{


// Static variables
string COpenCL::m_ClFilesPath;


// Helper functions
string LoadClFile(const string& Path);
cl::Program LoadClProgram(const cl::Context& context,  const string& Path, bool build);
cl::Platform findPlatform(const char * inPreferred);
string& string_tolower(string& str);

COpenCL::COpenCL(const char * PreferredPlatform, cl_device_type deviceType)
{
   if (PreferredPlatform == nullptr || string(PreferredPlatform) == "")
      m_Platform = cl::Platform::getDefault();
   else
      m_Platform = findPlatform(PreferredPlatform);

   // List devices for this platform
   vector<cl::Device> devices;
   m_Platform.getDevices(deviceType, &devices);

   if (devices.empty())
      throw cl::Error(CL_DEVICE_NOT_FOUND, "no default OpenCL device found");

   m_Device = devices[0];

   m_Context = cl::Context(m_Device);

   m_Queue = cl::CommandQueue(m_Context, devices[0]);

   m_Converter = make_shared<Conversions>(*this); 
}


void COpenCL::SetClFilesPath(const char * Path) 
{
   m_ClFilesPath = Path;
   if (m_ClFilesPath[m_ClFilesPath.size() - 1] != '\\' && 
      m_ClFilesPath[m_ClFilesPath.size() - 1] != '/')
   {
      // Missing trailing slash
      m_ClFilesPath += "/";
   }

}

const string& COpenCL::GetClFilePath()
{
   return m_ClFilesPath;
}

cl::CommandQueue& COpenCL::GetQueue()
{
   return m_Queue;
}

Conversions& COpenCL::GetConverter()
{
   return *m_Converter;
}

COpenCL::operator cl::Context& ()
{
   return m_Context;
}

COpenCL::operator cl::CommandQueue& ()
{
   return m_Queue;
}

COpenCL::operator cl::Device& ()
{
   return m_Device;
}

COpenCL::operator cl_context ()
{
   return m_Context();
}

COpenCL::operator cl_command_queue ()
{
   return m_Queue();
}

COpenCL::operator cl_device_id ()
{
   return m_Device();
}

bool COpenCL::SupportsNoCopy() const
{
   // TODO : Test all common devices to see if they support using the same
   // memory region for the host and the device.
   // Main candidates are : Intel GPU, AMD CPU, AMD Integrated GPU
   return false;
}

std::string COpenCL::GetDeviceName() const
{
   return m_Device.getInfo<CL_DEVICE_NAME>();
}

cl_device_type COpenCL::GetDeviceType() const
{
   return m_Device.getInfo<CL_DEVICE_TYPE>();
}

COpenCL::EPlatformType COpenCL::GetPlatformType() const
{
   string name = m_Platform.getInfo<CL_PLATFORM_NAME>();
   string_tolower(name);

   if (name.find("intel") != string::npos)
      return IntelPlatform;

   if (name.find("nvidia") != string::npos)
      return NvidiaPlatform;

   if (name.find("amd") != string::npos)
      return AmdPlatform;

   return OtherPlatform;
}

const char * COpenCL::ErrorName(cl_int status)
{
   switch (status)
   {
   case CL_SUCCESS:
      return "SUCCESS";
   case CL_BUILD_PROGRAM_FAILURE:
      return "CL_BUILD_PROGRAM_FAILURE";
   case CL_COMPILER_NOT_AVAILABLE:
      return "CL_COMPILER_NOT_AVAILABLE";
   case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
   case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
   case CL_IMAGE_FORMAT_MISMATCH:
      return "CL_IMAGE_FORMAT_MISMATCH";
   case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
   case CL_INVALID_ARG_INDEX:
      return "CL_INVALID_ARG_INDEX";
   case CL_INVALID_ARG_SIZE:
      return "CL_INVALID_ARG_SIZE";
   case CL_INVALID_ARG_VALUE:
      return "CL_INVALID_ARG_VALUE";
   case CL_INVALID_BINARY:
      return "CL_INVALID_BINARY";
   case CL_INVALID_BUFFER_SIZE:
      return "CL_INVALID_BUFFER_SIZE";
   case CL_INVALID_BUILD_OPTIONS:
      return "CL_INVALID_BUILD_OPTIONS";
   case CL_INVALID_COMMAND_QUEUE:
      return "CL_INVALID_COMMAND_QUEUE";
   case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
   case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
   case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
   case CL_INVALID_EVENT:
      return "CL_INVALID_EVENT";
   case CL_INVALID_EVENT_WAIT_LIST:
      return "CL_INVALID_EVENT_WAIT_LIST";
   case CL_INVALID_GL_OBJECT:
      return "CL_INVALID_GL_OBJECT";
   case CL_INVALID_GLOBAL_OFFSET:
      return "CL_INVALID_GLOBAL_OFFSET";
   case CL_INVALID_HOST_PTR:
      return "CL_INVALID_HOST_PTR";
   case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
   case CL_INVALID_IMAGE_SIZE:
      return "CL_INVALID_IMAGE_SIZE";
   case CL_INVALID_KERNEL_NAME:
      return "CL_INVALID_KERNEL_NAME";
   case CL_INVALID_KERNEL:
      return "CL_INVALID_KERNEL";
   case CL_INVALID_KERNEL_ARGS:
      return "CL_INVALID_KERNEL_ARGS";
   case CL_INVALID_KERNEL_DEFINITION:
      return "CL_INVALID_KERNEL_DEFINITION";
   case CL_INVALID_MEM_OBJECT:
      return "CL_INVALID_MEM_OBJECT";
   case CL_INVALID_OPERATION:
      return "CL_INVALID_OPERATION";
   case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
   case CL_INVALID_PROGRAM:
      return "CL_INVALID_PROGRAM";
   case CL_INVALID_PROGRAM_EXECUTABLE:
      return "CL_INVALID_PROGRAM_EXECUTABLE";
   case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
   case CL_INVALID_SAMPLER:
      return "CL_INVALID_SAMPLER";
   case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
   case CL_INVALID_WORK_DIMENSION:
      return "CL_INVALID_WORK_DIMENSION";
   case CL_INVALID_WORK_GROUP_SIZE:
      return "CL_INVALID_WORK_GROUP_SIZE";
   case CL_INVALID_WORK_ITEM_SIZE:
      return "CL_INVALID_WORK_ITEM_SIZE";
   case CL_MAP_FAILURE:
      return "CL_MAP_FAILURE";
   case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
   case CL_MEM_COPY_OVERLAP:
      return "CL_MEM_COPY_OVERLAP";
   case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
   case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
   case CL_PROFILING_INFO_NOT_AVAILABLE:
      return "CL_PROFILING_INFO_NOT_AVAILABLE";
   default:
      return "Unknown CL error";
   }

}

// Helper functions

string& string_tolower(string& str)
{
   for (char& c : str)
      c = (char) tolower(c);

   return str;
}

cl::Platform findPlatform(const char * inPreferred)
{
   string preferred = inPreferred;

   string_tolower(preferred);

   vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);

   for (auto platform : platforms)
   {
      string name;
      platform.getInfo(CL_PLATFORM_NAME, &name);

      if (string_tolower(name).find(preferred) != string::npos)
         return platform;
   }

   // Not found - use default platform
   return cl::Platform::getDefault();
}

}
