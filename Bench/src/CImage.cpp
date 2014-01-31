////////////////////////////////////////////////////////////////////////////////
//! @file	: CImage.cpp
//! @date   : Jul 2013
//!
//! @brief  : Implementation of a simple image class
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

#include <cstddef>
#include "CImage.h"

uint CalculateStep(uint W, uint C, uint D, uint Align);
template<class T> size_t align(T& Val, size_t Align);   // Returns the offset (amount of change applied to Val)

CSimpleImage::CSimpleImage()
: m_Data(nullptr),
  m_Offset(0)
{
   SImage Img = {0};
   ((SImage&)*this) = Img;
}

CSimpleImage::CSimpleImage(const SImage& Img)
: m_Data(nullptr),
  m_Offset(0)
{
   Create(Img);
}

CSimpleImage::~CSimpleImage()
{
   Free();
}

uint Depth(SImage::EDataType Type)
{
   switch (Type)
   {
   case SImage::U8:
   case SImage::S8:
      return 8;
   case SImage::U16:
   case SImage::S16:
      return 16;
   case SImage::U32:
   case SImage::S32:
   case SImage::F32:
      return 32;
   }

   return 0;
}

uint CSimpleImage::Depth() const
{
   return ::Depth(Type);
}

uint CSimpleImage::BytesPerPixel() const
{
   return Depth() * Channels / 8;
}

uint CSimpleImage::BytesWidth() const
{
   return Width * BytesPerPixel();
}

unsigned char * CSimpleImage::Data(uint Row)
{
   return m_Data + Step * Row;
}

const unsigned char * CSimpleImage::Data(uint Row) const
{
   return m_Data + Step * Row;
}

unsigned char * CSimpleImage::Data(uint x, uint y)
{
   return m_Data + Step * y + x * BytesPerPixel();
}

const unsigned char * CSimpleImage::Data(uint x, uint y) const
{
   return m_Data + Step * y + x * BytesPerPixel();
}

void CSimpleImage::Create(const SImage& Img, uint Align)
{
   ((SImage&)*this) = Img;
   Free();
   m_Data = new unsigned char[Step * Height + Align];
   m_Offset = (uint) align(m_Data, Align);
}

void CSimpleImage::Create(uint W, uint H, uint C, SImage::EDataType T, uint Align)
{
   SImage Img = {W, H, CalculateStep(W, C, ::Depth(T), Align), C, T};
   Create(Img);
}

void CSimpleImage::Free()
{
   delete [] (m_Data - m_Offset);
   m_Data = nullptr;
   m_Offset = 0;
}

const SImage& CSimpleImage::ToSImage() const
{
   return *this;
}

CImageROI::CImageROI(CSimpleImage& Image, uint X, uint Y, uint W, uint H)
{
   ((SImage&)*this) = Image.ToSImage();
   m_Data = Image.Data(X, Y);
   Width = W;
   Height = H;
}


// Helpers

template<class T>
size_t align(T& Val, size_t Align)
{
   size_t Mod = size_t(Val) % Align;
   if (Mod == 0)
      return 0;

   size_t Offset = Align - Mod;

   Val += Offset;
   return Offset;
}

uint CalculateStep(uint W, uint C, uint D, uint Align)
{
   size_t Bytes = W * D * C / 8;
   align(Bytes, Align);
   return (uint) Bytes;
}
