////////////////////////////////////////////////////////////////////////////////
//! @file	: OpenCL.h 
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

#pragma once


#include "Basic.h"


// Include C interface of OpenCL
// NVIDIA does not currently support OpenCL 1.2 in its drivers
// So we force usage of the OpenCL 1.1 API
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <CL/opencl.h>
#undef CL_VERSION_1_2


// We are working with exceptions instead of having error checking on every cl api call
#define __CL_ENABLE_EXCEPTIONS

// Include C++ interface of OpenCL
#include <CL/cl.hpp>


#include <memory>

namespace OpenCLIPP
{

class Conversions;

/// Takes care of initializing OpenCL
/// Contains an OpenCL Device, Context and CommandQueue
/// An instance of this object is needed to create most other object of the library
class CL_API COpenCL
{
public:

   /// Initializes OpenCL.
   /// \param PreferredPlatform : Can be set to a specific platform (Ex: "Intel") and
   ///         that platform will be used if available. If the preferred platform is not
   ///         found or is not specified, the default OpenCL platform will be used.
   /// \param deviceType : can be used to specicy usage of a device type (Ex: CL_DEVICE_TYPE_GPU)
   ///         See cl_device_type for allowed values
   COpenCL(const char * PreferredPlatform = "", cl_device_type deviceType = CL_DEVICE_TYPE_ALL);

   /// Returns the name of the OpenCL device
   std::string GetDeviceName() const;

   /// Returns the type of the OpenCL device
   /// Can be CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU or other types
   cl_device_type GetDeviceType() const;

   /// Lists the common OpenCL platforms
   enum EPlatformType
   {
      NvidiaPlatform,
      AmdPlatform,
      IntelPlatform,
      OtherPlatform,
   };

   /// Returns the type of the OpenCL platform
   EPlatformType GetPlatformType() const;

   /// Returns the name of the OpenCL error
   /// \param status : An OpenCL error code
   /// \return the name of the given error code
   static const char * ErrorName(cl_int status);

   /// Tells COpenCL where the .cl files are located.
   /// It must be called with the full path of "OpenCLIPP/cl-files/".
   /// .cl file location must be specified before creating any program.
   /// The path must not contain spaces
   /// \param Path : Full path where the .cl files are located
   static void SetClFilesPath(const char * Path);

   /// Returns the .cl files path (for internal use)
   /// \return the path where the .cl files are located
   static const std::string& GetClFilePath();

   /// Returns the OpenCL CommandQueue (for internal use)
   cl::CommandQueue& GetQueue();

   /// Returns the color image converter program (for internal use)
   Conversions& GetConverter();

   operator cl::Context& ();        ///< Converts to a cl::Context
   operator cl::CommandQueue& ();   ///< Converts to a cl::CommandQueue
   operator cl::Device& ();         ///< Converts to a cl::Device

   operator cl_context ();          ///< Converts to a cl_context
   operator cl_command_queue ();    ///< Converts to a cl_command_queue
   operator cl_device_id ();        ///< Converts to a cl_device_id


   /// No-copy support on the OpenCL device.
   /// True when the device uses the same memory for the host and the device.
   /// When true, image transfers to the device are instantaneous.
   bool SupportsNoCopy() const;

protected:
   cl::Platform m_Platform;      ///< The OpenCL platform (like nVidia)
   cl::Device m_Device;          ///< The OpenCL device (like GTX 680)
   cl::Context m_Context;        ///< The OpenCL context for the device
   cl::CommandQueue m_Queue;     ///< The OpenCL command queue for the context

   std::shared_ptr<Conversions> m_Converter;  ///< Instance of the Conversions program, needed by ColorImage class

   static std::string m_ClFilesPath;   ///< Path to the .cl files
};

}  // End of namespace
