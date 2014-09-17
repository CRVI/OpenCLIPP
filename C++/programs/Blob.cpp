////////////////////////////////////////////////////////////////////////////////
//! @file	: Blob.cpp
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

#include "Programs/Blob.h"

#include "kernel_helpers.h"


namespace OpenCLIPP
{

Blob::Blob(COpenCL& CL)
:  ImageProgram(CL, "Blob.cl"),
   m_InfoBuffer(CL, &m_BlobInfo, 1)
{ }

void Blob::SBlobInfo::Init(int connectType)
{
   ConnectType = connectType;	// Must be 4 or 8
   NbBlobs = 0;
   LastUsefulIteration = 2;   // Start with two iterations
}

void Blob::PrepareFor(ImageBase& Source)
{
   if (m_TempBuffer == nullptr || Source.Width() > m_TempBuffer->Width() || Source.Height() > m_TempBuffer->Height())
      m_TempBuffer = std::make_shared<TempImage>(*m_CL, Source.ImageSize(), SImage::S32);
}

void Blob::ComputeLabels(Image& Source, Image& Labels, int ConnectType)
{
   if (ConnectType != 4 && ConnectType != 8)
      throw cl::Error(CL_INVALID_VALUE, "Wrong connect type in Blob::ComputeLabels");

   if (Labels.Depth() != 32 || Labels.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "Wrong Labels image type in Blob::ComputeLabels - Labels must be 32 bit integer");

   PrepareFor(Source);

   CheckSameSize(Source, Labels);
   CheckCompatibility(Labels, *m_TempBuffer);

   m_BlobInfo.Init(ConnectType);

   m_InfoBuffer.Send();

   // Initialize the label image
   Kernel(init_label, Source, Out(Labels, *m_TempBuffer), Source.Step(), Labels.Step(), m_TempBuffer->Step(), m_InfoBuffer);

   // These two labeling steps need to be executed at least twice each
   int i = 0;
   while (i <= m_BlobInfo.LastUsefulIteration)
   {
      i++;

      Kernel(label_step1, Labels, *m_TempBuffer, Labels.Step(), m_TempBuffer->Step(), m_InfoBuffer, i);
      Kernel(label_step2, Labels, *m_TempBuffer, Labels.Step(), m_TempBuffer->Step(), m_InfoBuffer, i);

      if (i >= 2)
         m_InfoBuffer.Read(true);
   }

}

void Blob::RenameLabels(Image& Labels)
{
   if (Labels.Depth() != 32 || Labels.IsFloat())
      throw cl::Error(CL_INVALID_VALUE, "Wrong Labels image type in Blob::RenameLabels - Labels must be 32 bit integer");

   if (m_TempBuffer == nullptr)
      throw cl::Error(CL_INVALID_MEM_OBJECT, "ComputeLabels must be called before renaming the labels");

   CheckCompatibility(Labels, *m_TempBuffer);

   // Rename the labels
   Kernel(reorder_labels1, Labels, *m_TempBuffer, Labels.Step(), m_TempBuffer->Step(), m_InfoBuffer);
   Kernel(reorder_labels2, Labels, *m_TempBuffer, Labels.Step(), m_TempBuffer->Step(), m_InfoBuffer);

   m_InfoBuffer.Read(true);
}

}
