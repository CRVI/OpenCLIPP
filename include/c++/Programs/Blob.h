////////////////////////////////////////////////////////////////////////////////
//! @file	: Blob.h
//! @date   : Jul 2013
//!
//! @brief  : Blob labeling
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

#include "Program.h"

namespace OpenCLIPP
{

/// A program that does Blob labeling on an image
class CL_API Blob : public ImageProgram
{
public:
   Blob(COpenCL& CL);

   struct SBlobInfo
   {
      void Init(int connectType = 4);

      int ConnectType;	   // 4 or 8
      int NbBlobs;
      int LastUsefulIteration;
   };

   void PrepareFor(ImageBase& Source);  ///< Allocate internal temporary buffer and build the program

   /// Compute the blob labels for the given image.
   /// PrepareFor() must be called with the same Source image before calling ComputeLabels()
   /// All non-zero pixels will be grouped with their neighbours and given a label number
   /// After calling, Labels image buffer will contain the label values for each pixel,
   /// and -1 (or 0xffffffff) for pixels that were 0
   /// \param Source : The image to analyze
   /// \param Labels : must be a 32b integer image
   /// \param ConnectType : Type of pixel connectivity, can be 4 or 8
   void ComputeLabels(Image& Source, Image& Labels, int ConnectType = 4);

   /// Renames the labels to be from 0 to NbLabels-1.
   /// \param Labels : must be an image resulting from a previous call to ComputeLabels()
   void RenameLabels(Image& Labels);

protected:

   std::shared_ptr<TempImage> m_TempBuffer;

   SBlobInfo m_BlobInfo;
   Buffer m_InfoBuffer;

   void operator = (const Blob&) { }  // Not a copyable object - because of m_InfoBuffer
};

}
