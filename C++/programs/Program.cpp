////////////////////////////////////////////////////////////////////////////////
//! @file	: Program.cpp
//! @date   : Jul 2013
//!
//! @brief  : Objects representing OpenCL programs
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

#include "Programs/Program.h"
#include <iostream>
#include <fstream>

using namespace std;

namespace OpenCLIPP
{

const char ** GetDataTypeList();

Program::Program(COpenCL& CL, const char * Path, const char * options)
:  m_CL(&CL),
   m_Path(Path),
   m_Options(options),
   m_Built(false)
{ }

Program::Program(COpenCL& CL, bool, const char * Source, const char * options)
:  m_CL(&CL),
   m_Source(Source),
   m_Options(options),
   m_Built(false)
{ }

#ifndef _MSC_VER
#define IsDebuggerPresent() false
#endif // _MSC_VER

bool Program::Build()
{
   if (m_Built)
      return true;

   string Path, Source = m_Source;

   if (Source == "")
   {
      if (m_Path == "")
         return false;

      Path = m_CL->GetClFilePath() + m_Path;

      Source = LoadClFile(Path);
   }

   m_Program = cl::Program(*m_CL, Source, false);

   string optionStr = m_Options;
      
   if (m_Path != "" && IsDebuggerPresent() &&
      m_CL->GetPlatformType() == COpenCL::IntelPlatform && m_CL->GetDeviceType() == CL_DEVICE_TYPE_CPU)
   {
      // Add debug information for Intel OpenCL SDK Debugger
      optionStr += " -g -s \"" + Path + "\"";
   }


   // Add include path
   optionStr += " -I " + m_CL->GetClFilePath();    // The path must not have spaces if using NVIDIA platform

   // Try to build
   try
   {
      m_Program.build(optionStr.c_str());
      m_Built = true;
   }
   catch (cl::Error error)
   {
      // Build failed

      string build_info;
      m_Program.getBuildInfo(*m_CL, CL_PROGRAM_BUILD_LOG, &build_info);

      cerr << "Program build failed : " << build_info << endl;
      if (Path != "")
         cerr << " - in file : " << Path << endl;

      throw error;   // Rethrow
   }

   return true;
}

std::string Program::LoadClFile(const std::string& Path)
{
   ifstream file(Path);

   if (!file.is_open())
      throw cl::Error(CL_INVALID_PROGRAM, "could not find the program file");

   string content(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));

   return content;
}


// MultiProgram
MultiProgram::MultiProgram(COpenCL& CL)
:  m_CL(&CL)
{ }

void MultiProgram::SetProgramInfo(const char * Path, const vector<string>& Options)
{
   SetProgramInfo(false, "", Options, Path);
}

void MultiProgram::SetProgramInfo(bool /*fromSource*/, const char * Source, const vector<string>& Options, const char * Path)
{
   m_Programs.assign(Options.size(), nullptr);

   for (size_t i = 0; i < Options.size(); i++)
      if (string(Source) != "")
         m_Programs[i] = make_shared<Program>(*m_CL, true, Source, Options[i].c_str());
      else
         m_Programs[i] = make_shared<Program>(*m_CL, Path, Options[i].c_str());
}

Program& MultiProgram::GetProgram(uint Id)
{
   assert(Id < m_Programs.size());
   m_Programs[Id]->Build();   // Build if needed
   return *m_Programs[Id];
}

void MultiProgram::PrepareProgram(uint Id)
{
   assert(Id < m_Programs.size());
   m_Programs[Id]->Build();
}


// ImageProgram
ImageProgram::ImageProgram(COpenCL& CL, const char * Path)
:  MultiProgram(CL)
{
   const vector<string> Options = GetOptions();

   SetProgramInfo(Path, Options);
}

ImageProgram::ImageProgram(COpenCL& CL, bool fromSource, const char * Source)
:  MultiProgram(CL)
{
   const vector<string> Options = GetOptions();

   SetProgramInfo(fromSource, Source, Options);
}

const vector<string> ImageProgram::GetOptions()
{
   vector<string> Options(SImage::NbDataTypes * MaxNbChannels);

   for (int i = 0; i < SImage::NbDataTypes; i++)
   {
      SImage::EDataType Type = SImage::EDataType(i);

      string options = string("-D ") + GetDataTypeList()[i];

      Options[GetProgramId(Type, 1)] = options;
      for (int j = 2; j <= MaxNbChannels; j++)
         Options[GetProgramId(Type, j)] = options + " -D NBCHAN=" + to_string(j);
   }

   return Options;
}

void ImageProgram::PrepareFor(const ImageBase& Source)
{
   SelectProgram(Source).Build();
}

Program& ImageProgram::SelectProgram(const ImageBase& Img1)
{
   if (Img1.DataType() < 0 || Img1.DataType() >= SImage::NbDataTypes)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "unsupported image format used with ImageProgram");

   if (Img1.NbChannels() < 1 || Img1.NbChannels() > MaxNbChannels)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "unsupported number of channels");

   return GetProgram(GetProgramId(Img1.DataType(), Img1.NbChannels()));
}

Program& ImageProgram::SelectProgram(const ImageBase& Img1, const ImageBase&)
{
   // Only Img1 is used to select the program
   return SelectProgram(Img1);
}

Program& ImageProgram::SelectProgram(const ImageBase& Img1, const ImageBase&, const ImageBase&)
{
   // Only Img1 is used to select the program
   return SelectProgram(Img1);
}

Program& ImageProgram::SelectProgram(const ImageBase& Img1, const ImageBase&, const ImageBase&, const ImageBase&)
{
   // Only Img1 is used to select the program
   return SelectProgram(Img1);
}

uint ImageProgram::GetProgramId(SImage::EDataType Type, uint NbChannels)
{
   return (NbChannels - 1) * SImage::NbDataTypes + Type;
}


// VectorProgram
VectorProgram::VectorProgram(COpenCL& CL, const char * Path)
:  MultiProgram(CL)
{
   const vector<string> Options = GetOptions();

   SetProgramInfo(Path, Options);
}

VectorProgram::VectorProgram(COpenCL& CL, bool fromSource, const char * Source)
:  MultiProgram(CL)
{
   const vector<string> Options = GetOptions();

   SetProgramInfo(fromSource, Source, Options);
}

uint VectorProgram::GetProgramId(SImage::EDataType Type, EProgramVersions Version)
{
   return Version * SImage::NbDataTypes + Type;
}

const vector<string> VectorProgram::GetOptions()
{
   vector<string> Options(SImage::NbDataTypes * NbVersions);

   for (int i = 0; i < SImage::NbDataTypes; i++)
   {
      SImage::EDataType Type = SImage::EDataType(i);

      int VectorWidth = GetVectorWidth(Type);

      string options = string("-D VEC_WIDTH=") + to_string(VectorWidth) + " -D " + GetDataTypeList()[i];

      Options[GetProgramId(Type, Fast)]      = options;
      Options[GetProgramId(Type, Standard)]  = options + " -D WITH_PADDING";
      Options[GetProgramId(Type, Unaligned)] = options + " -D WITH_PADDING -D UNALIGNED";
   }

   return Options;
}

int VectorProgram::GetVectorWidth(SImage::EDataType Type)
{
   // NOTE : This could be customized depending on the hardware for best performance
   switch (Type)
   {
   case SImage::U8:
   case SImage::S8:
      return 8;
   case SImage::U16:
   case SImage::S16:
      return 4;
   case SImage::U32:
   case SImage::S32:
   case SImage::F32:
   case SImage::F64:
      return 2;
   case SImage::NbDataTypes:
   default:
      assert(false); // wrong enum value
      return 2;
   }

}

bool VectorProgram::IsImageFlush(const ImageBase& Source)
{
   if (Source.Width() * Source.NbChannels() * Source.DepthBytes() != Source.Step())
      return false;  // Image has padding

   if ((Source.Width() * Source.NbChannels()) % GetVectorWidth(Source.DataType()) != 0)
      return false;  // width is not a multiple of VectorWidth

   return true;
}

bool VectorProgram::IsImageAligned(const ImageBase& Source)
{
   uint BytesPerWorker = Source.DepthBytes() * Source.NbChannels() * GetVectorWidth(Source.DataType());
   return (Source.Step() % BytesPerWorker == 0);
}

cl::NDRange VectorProgram::GetRange(EProgramVersions Version, const ImageBase& Img1)
{
   switch (Version)
   {
   case Fast:
      // The fast version uses a 1D range
      return cl::NDRange(Img1.Width() * Img1.Height() * Img1.NbChannels() / GetVectorWidth(Img1.DataType()), 1, 1);

   case Standard:
   case Unaligned:
      // The other versions use a 2D range
      return Img1.VectorRange(GetVectorWidth(Img1.DataType()));

   case NbVersions:
   default:
      throw cl::Error(CL_INVALID_PROGRAM, "Invalid program version in VectorProgram");
   }

}

cl::NDRange VectorProgram::GetRange(const ImageBase& Img1)
{
   return GetRange(SelectVersion(Img1), Img1);
}

cl::NDRange VectorProgram::GetRange(const ImageBase& Img1, const ImageBase& Img2)
{
   return GetRange(SelectVersion(Img1, Img2), Img1);
}

cl::NDRange VectorProgram::GetRange(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3)
{
  return GetRange(SelectVersion(Img1, Img2, Img3), Img1);
}

cl::NDRange VectorProgram::GetRange(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4)
{
   return GetRange(SelectVersion(Img1, Img2, Img3, Img4), Img1);
}

Program& VectorProgram::GetProgram(SImage::EDataType Type, EProgramVersions Version)
{
   return MultiProgram::GetProgram(GetProgramId(Type, Version));
}

void VectorProgram::PrepareFor(const ImageBase& Source)
{
   SelectProgram(Source).Build();
}

VectorProgram::EProgramVersions VectorProgram::SelectVersion(const ImageBase& Img1)
{
   if (!IsImageAligned(Img1))
      return Unaligned;    // Use the slow unaligned version

   if (!IsImageFlush(Img1))
      return Standard;     // Use standard version

   return Fast;  // Use fast version
}

VectorProgram::EProgramVersions VectorProgram::SelectVersion(const ImageBase& Img1, const ImageBase& Img2)
{
   if (!IsImageAligned(Img1) || !IsImageAligned(Img2))
      return Unaligned;   // Use the slow unaligned version

   if (!IsImageFlush(Img2))
      return Standard;     // Use standard version

   return SelectVersion(Img1);
}

VectorProgram::EProgramVersions VectorProgram::SelectVersion(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3)
{
   if (!IsImageAligned(Img1) || !IsImageAligned(Img2) || !IsImageAligned(Img3))
      return Unaligned;   // Use the slow unaligned version

   if (!IsImageFlush(Img3))
      return Standard;     // Use standard version

   return SelectVersion(Img1, Img2);
}

VectorProgram::EProgramVersions VectorProgram::SelectVersion(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4)
{
   if (!IsImageAligned(Img1) || !IsImageAligned(Img2) || !IsImageAligned(Img3) || !IsImageAligned(Img4))
      return Unaligned;   // Use the slow unaligned version

   if (!IsImageFlush(Img4))
      return Standard;     // Use standard version

   return SelectVersion(Img1, Img2, Img3);
}

Program& VectorProgram::SelectProgram(const ImageBase& Img1)
{
   return GetProgram(Img1.DataType(), SelectVersion(Img1));
}

Program& VectorProgram::SelectProgram(const ImageBase& Img1, const ImageBase& Img2)
{
   return GetProgram(Img1.DataType(), SelectVersion(Img1, Img2));
}

Program& VectorProgram::SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3)
{
   return GetProgram(Img1.DataType(), SelectVersion(Img1, Img2, Img3));
}

Program& VectorProgram::SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4)
{
   return GetProgram(Img1.DataType(), SelectVersion(Img1, Img2, Img3, Img4));
}


// Helper functions for programs

const char ** GetDataTypeList()
{
   static const char * PixelTypes[SImage::NbDataTypes] = {"U8", "S8", "U16", "S16", "U32", "S32", "F32", "F64"};    // Keep in synch with EPixelTypes
   return PixelTypes;
}

bool SameType(const ImageBase& Img1, const ImageBase& Img2)
{
   return (Img1.IsFloat() == Img2.IsFloat() &&
      Img1.IsUnsigned() == Img2.IsUnsigned() &&
      Img1.Depth() == Img2.Depth() &&
      Img1.NbChannels() == Img2.NbChannels());
}

void CheckSameSize(const ImageBase& Img1, const ImageBase& Img2)
{
   if (Img1.Width() != Img2.Width() || Img1.Height() != Img2.Height())
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "Different image sizes used");
}

void CheckCompatibility(const ImageBase& Img1, const ImageBase& Img2)
{
   CheckSameSize(Img1, Img2);

   if (Img1.IsFloat() != Img2.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   if (Img1.IsUnsigned() != Img2.IsUnsigned())
     throw cl::Error(CL_INVALID_VALUE, "Different image types used");
}

void CheckSizeAndType(const ImageBase& Img1, const ImageBase& Img2)
{
   CheckCompatibility(Img1, Img2);

   if (Img1.Depth() != Img2.Depth())
      throw cl::Error(CL_INVALID_VALUE, "Different image depth used");

   if (Img1.IsUnsigned() != Img2.IsUnsigned())
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");
}

void CheckSameNbChannels(const ImageBase& Img1, const ImageBase& Img2)
{
   if (Img1.NbChannels() != Img2.NbChannels())
      throw cl::Error(CL_INVALID_VALUE, "Different number of channels used");
}

void CheckSimilarity(const ImageBase& Img1, const ImageBase& Img2)
{
   CheckSizeAndType(Img1, Img2);

   CheckSameNbChannels(Img1, Img2);
}

void CheckFloat(const ImageBase& Img)
{
   if (!Img.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "non-float image used when not allowed");
}

void CheckNotFloat(const ImageBase& Img)
{
   if (Img.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "float image used when not allowed");
}

void Check1Channel(const ImageBase& Img)
{
   if (Img.NbChannels() > 1)
      throw cl::Error(CL_INVALID_VALUE, "only 1 channel images are allowed");
}

}
