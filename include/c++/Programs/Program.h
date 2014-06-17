////////////////////////////////////////////////////////////////////////////////
//! @file	: Program.h
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

/// Programs operate either on Image objects or on Image objects
/// Some programs are available in both standard (Image) and Vector (Image) versions.
/// For those programs, the Vector version is usually substantially faster as it will use vector processing

#pragma once

#include "../OpenCL.h"
#include "../Image.h"

namespace OpenCLIPP
{

/// Creates a cl::Program from a .cl file (or from source code)
class CL_API Program
{
public:

   /// Constructor - from .cl file.
   /// Program is not built automatically, Build() must be called before using the program.
   /// \param CL : A COpenCL instance
   /// \param Path : Path of the .cl file - must be relative to the path given by COpenCL::SetClFilesPath()
   /// \param options : A string to give as options when invoking the OpenCL C compiler (useful to specify defines)
   Program(COpenCL& CL, const char * Path, const char * options = "");

   /// Constructor - from source code.
   /// Program is not built automatically, Build() must be called before using the program.
   /// \param CL : A COpenCL instance
   /// \param FromSource : Indicates that source code is directly specified (instead of using a .cl file)
   /// \param Source : OpenCL C source code of the program
   /// \param options : A string to give as options when invoking the OpenCL C compiler (useful to specify defines)
   Program(COpenCL& CL, bool FromSource, const char * Source, const char * options = "");

   /// Destructor.
   virtual ~Program() { }

   /// Builds the OpenCL program.
   /// Uses either the source or the content of the .cl file given to one of the constructors
   /// as content of the program.
   /// Compiler errors are dealt in the following way : 
   ///  - A cl::Error exception is thrown
   ///  - Build messages are sent to std::cerr
   /// Take note that building an OpenCL C program can be long (sometimes >100ms).
   /// So build the program in advance if a result is desired quickly.
   /// \return true if build operation is succesful
   bool Build();

   operator cl::Program& ()
   {
      return m_Program;
   }

protected:
   COpenCL * m_CL;            ///< Pointer to the COpenCL object this program is assotiated to
   std::string m_Path;        ///< Path of the .cl file
   std::string m_Source;      ///< Given source of the program
   std::string m_Options;     ///< Options to give to the OpenCL C compiler
   cl::Program m_Program;     ///< The ecapsulated program object
   bool m_Built;              ///< true when the program has been successfully built

   static std::string LoadClFile(const std::string& Path);   ///< Reads the content of the given file
};


/// Holder of multiple Program (not useable directly)
class CL_API MultiProgram
{
public:
   virtual ~MultiProgram() { }   ///< Destructor

   COpenCL& GetCL() { return *m_CL; }  ///< Returns a reference to the COpenCL object this program is assotiated to

protected:
   MultiProgram(COpenCL& CL);    ///< Constructor

   /// Sets the information about a .cl program file
   /// \param Path : Path of the .cl file - must be relative to the path given by COpenCL::SetClFilesPath()
   /// \param Options : A list containing the options to give when building the different versions of the program
   void SetProgramInfo(const char * Path, const std::vector<std::string>& Options);

   /// Sets the information about a program
   /// \param fromSource : Indicates wether a .cl file is used or if source code is supplied
   /// \param Source : Source code (used if fromSource is true)
   /// \param Options : A list containing the options to give when building the different versions of the program
   /// \param Path : Path of the .cl file - must be relative to the path given by COpenCL::SetClFilesPath().
   ///   used if fromSource is false
   void SetProgramInfo(bool fromSource, const char * Source,
      const std::vector<std::string>& Options, const char * Path = "");

   Program& GetProgram(uint Id);    ///< Builds the program specified by Id and returns a reference to it

   void PrepareProgram(uint Id);    ///< Builds the program specified by Id

   COpenCL * m_CL;   ///< Pointer to the COpenCL object this program is assotiated to

private:
   typedef std::shared_ptr<Program> ProgramPtr;

   std::vector<ProgramPtr> m_Programs;
};


/// A program that can operate on all supported data types.
/// Contains a version of the program for each data type : S8, U8, S16, U16, S32, U32, F32, F64
class CL_API ImageProgram : public MultiProgram
{
public:

   /// Initialize the program with a .cl file.
   /// Program is not built by the constructor, it will be built when needed.
   /// Call PrepareFor() to have the program ready for later use.
   /// \param CL : A COpenCL instance
   /// \param Path : Path of the .cl file - must be relative to the path given by COpenCL::SetClFilesPath()
   ImageProgram(COpenCL& CL, const char * Path);

   /// Initialize the program with source code.
   /// Program is not built by the constructor, it will be built when needed.
   /// Call PrepareFor() to have the program ready for later use.
   /// \param CL : A COpenCL instance
   /// \param fromSource : Indicates that source code is directly specified (instead of using a .cl file)
   /// \param Source : OpenCL C source code of the program
   ImageProgram(COpenCL& CL, bool fromSource, const char * Source);

   /// Build the version of the program appropriate for this image.
   /// Building can take a lot of time (100+ms) so it is better to build
   /// the program during when starting so it will be ready when needed.
   void PrepareFor(const ImageBase& Source);

   /// Selects the appropriate program version for this image.
   /// Also builds the program version if it was not already built.
   Program& SelectProgram(const ImageBase& Img1);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4);

   static uint GetProgramId(SImage::EDataType Type, uint NbChannels);

   const static int MaxNbChannels = 4;

private:
   static const std::vector<std::string> GetOptions();
};


/// A program that uses vector operations for more troughput.
/// Contains three versions of the program for each data type : S8, U8, S16, U16, S32, U32, F32, F64
/// And one fast version that can operate on "flush" images with no padding and a width that is a multiple of the vector width
/// One standard version that can operate on images of any sizes and on images with padding, but with a step that is a multiple of the vector width
/// And one slower unaligned version that operates on any images.
class CL_API VectorProgram : public MultiProgram
{
public:

   /// Lists the three program versions
   enum EProgramVersions
   {
      Fast,          // Fast version for "flush" images
      Standard,      // Standard version
      Unaligned,     // Slow version for unaligned images
      NbVersions,
   };

   /// Initialize the program with a .cl file.
   /// Program is not built by the constructor, it will be built when needed.
   /// Call PrepareFor() to have the program ready for later use.
   /// \param CL : A COpenCL instance
   /// \param Path : Path of the .cl file - must be relative to the path given by COpenCL::SetClFilesPath()
   VectorProgram(COpenCL& CL, const char * Path);

   /// Initialize the program with source code.
   /// Program is not built by the constructor, it will be built when needed.
   /// Call PrepareFor() to have the program ready for later use.
   /// \param CL : A COpenCL instance
   /// \param fromSource : Indicates that source code is directly specified (instead of using a .cl file)
   /// \param Source : OpenCL C source code of the program
   VectorProgram(COpenCL& CL, bool fromSource, const char * Source);

   /// Build the version of the program appropriate for this image.
   /// Building can take a lot of time (100+ms) so it is better to build
   /// the program during when starting so it will be ready when needed.
   void PrepareFor(const ImageBase& Source);

   /// Selects the appropriate program version for the images.
   /// Also builds the program version if it was not already built.
   Program& SelectProgram(const ImageBase& Img1);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3);
   Program& SelectProgram(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4);

   /// Selects the appropriate program version for the images.
   static EProgramVersions SelectVersion(const ImageBase& Img1);
   static EProgramVersions SelectVersion(const ImageBase& Img1, const ImageBase& Img2);
   static EProgramVersions SelectVersion(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3);
   static EProgramVersions SelectVersion(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4);

   /// Returns true if the image has no padding and has a width that is a multiple
   /// of the vector width
   static bool IsImageFlush(const ImageBase& Source);

   /// Returns true if the step of the image is a multiple of the vector width
   static bool IsImageAligned(const ImageBase& Source);

   static cl::NDRange GetRange(EProgramVersions Version, const ImageBase& Img1);
   static cl::NDRange GetRange(const ImageBase& Img1);
   static cl::NDRange GetRange(const ImageBase& Img1, const ImageBase& Img2);
   static cl::NDRange GetRange(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3);
   static cl::NDRange GetRange(const ImageBase& Img1, const ImageBase& Img2, const ImageBase& Img3, const ImageBase& Img4);

   /// Builds the program specified by Id and returns a reference to it
   Program& GetProgram(SImage::EDataType Type, EProgramVersions Version);

   static uint GetProgramId(SImage::EDataType Type, EProgramVersions Version);

   ///< Returns the vector operation width to use for the given data type
   static int GetVectorWidth(SImage::EDataType Type);

private:
   static const std::vector<std::string> GetOptions();
};

// Helper functions for programs
bool SameType(const ImageBase& Img1, const ImageBase& Img2);         ///< Returns true if both images are of the same type
void CheckFloat(const ImageBase& Img);                               ///< Checks that the image contains float, throws a cl::Error if not
void CheckNotFloat(const ImageBase& Img);                            ///< Checks that the image does not contains float, throws a cl::Error if not
void Check1Channel(const ImageBase& Img);                            ///< Checks that the image contains only 1 channel, throws a cl::Error if not
void CheckSameSize(const ImageBase& Img1, const ImageBase& Img2);    ///< Checks that both images have the same size, throws a cl::Error if not

/// Checks sizes + float/signed/unsigned, throws a cl::Error if not.
/// Checks that both images are of the same size and same type (float/signed/unsigned)
void CheckCompatibility(const ImageBase& Img1, const ImageBase& Img2);

/// Checks sizes + float/signed/unsigned + depth, throws a cl::Error if not.
/// Checks that both images are of the same size and same type (float/signed/unsigned & depth)
/// Useful for channel conversion kernels
void CheckSizeAndType(const ImageBase& Img1, const ImageBase& Img2);

/// Checks sizes + float/signed/unsigned + depth + nb channels, throws a cl::Error if not.
/// Checks that both images are of the same size and same type (float/signed/unsigned & depth) and have the same number of channels
void CheckSimilarity(const ImageBase& Img1, const ImageBase& Img2);

/// Checks nb channels, throws a cl::Error if not.
/// Checks that both images have the same number of channels
void CheckSameNbChannels(const ImageBase& Img1, const ImageBase& Img2);

}
