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
      
   if (m_Path != "" && m_CL->IsOnIntelCPU() && IsDebuggerPresent())
   {
      // Add debug information for Intel OpenCL SDK Debugger
      optionStr += " -g -s \"" + Path + "\"";
   }

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

void MultiProgram::SetProgramInfo(const char * Path, uint NbPrograms, const char ** Defines)
{
   SetProgramInfo(false, "", NbPrograms, Defines, Path);
}

void MultiProgram::SetProgramInfo(bool /*fromSource*/, const char * Source,
                                         uint NbPrograms, const char ** Defines, const char * Path)
{
   m_Programs.assign(NbPrograms, nullptr);

   for (uint i = 0; i < NbPrograms; i++)
   {
      string option = "-D ";
      option += Defines[i];
      if (string(Source) != "")
         m_Programs[i] = make_shared<Program>(*m_CL, true, Source, option.c_str());
      else
         m_Programs[i] = make_shared<Program>(*m_CL, Path, option.c_str());
   }

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
   const char * Defines[NbPixelTypes] = {"I", "UI", "F"};    // Keep in synch with EPixelTypes
   SetProgramInfo(Path, NbPixelTypes, Defines);
}

ImageProgram::ImageProgram(COpenCL& CL, bool fromSource, const char * Source)
:  MultiProgram(CL)
{
   const char * Defines[NbPixelTypes] = {"I", "UI", "F"};    // Keep in synch with EPixelTypes
   SetProgramInfo(fromSource, Source, NbPixelTypes, Defines);
}

void ImageProgram::PrepareFor(ImageBase& Source)
{
   SelectProgram(Source).Build();
}

Program& ImageProgram::SelectProgram(ImageBase& Source)
{
   if (Source.IsFloat())
      return GetProgram(Float);

   if (Source.IsUnsigned())
      return GetProgram(Unsigned);

   return GetProgram(Signed);
}


// ImageBufferProgram
ImageBufferProgram::ImageBufferProgram(COpenCL& CL, const char * Path)
:  MultiProgram(CL)
{
   const char * Defines[NbPixelTypes] = {"U8", "S8", "U16", "S16", "U32", "S32", "F32"};    // Keep in synch with EPixelTypes

   SetProgramInfo(Path, NbPixelTypes, Defines);
}

ImageBufferProgram::ImageBufferProgram(COpenCL& CL, bool fromSource, const char * Source)
:  MultiProgram(CL)
{
   const char * Defines[NbPixelTypes] = {"U8", "S8", "U16", "S16", "U32", "S32", "F32"};    // Keep in synch with EPixelTypes

   SetProgramInfo(fromSource, Source, NbPixelTypes, Defines);
}

void ImageBufferProgram::PrepareFor(ImageBase& Source)
{
   SelectProgram(Source).Build();
}

Program& ImageBufferProgram::SelectProgram(ImageBase& Source)
{
   if (Source.DataType() < 0 || Source.DataType() >= NbPixelTypes)
      throw cl::Error(CL_IMAGE_FORMAT_NOT_SUPPORTED, "unsupported image format used with ImageBufferProgram");

   return GetProgram(Source.DataType());
}


// Helper functions for programs

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

void CheckSimilarity(const ImageBase& Img1, const ImageBase& Img2)
{
   CheckSizeAndType(Img1, Img2);

   if (Img1.NbChannels() != Img2.NbChannels())
      throw cl::Error(CL_INVALID_VALUE, "Different number of channels used");
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

}
