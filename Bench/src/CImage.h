////////////////////////////////////////////////////////////////////////////////
//! @file	: CImage.h
//! @date   : Jul 2013
//!
//! @brief  : Declaration of a simple image class
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

typedef unsigned int uint;

#include <SImage.h>

struct SSize
{
   SSize() { }
   SSize(uint W, uint H)
   :  Width(W),
      Height(H)
   { }

   uint Width;
   uint Height;
};

struct SPoint
{
   SPoint() { }
   SPoint(uint x, uint y)
   :  X(x),
      Y(y)
   { }

   uint X;
   uint Y;
};

class CSimpleImage : public SImage
{
public:

   const static int DefaultAlignement = 8;

   CSimpleImage();
   CSimpleImage(const SImage& Img);
   virtual ~CSimpleImage();

   uint Depth() const;
   uint BytesPerPixel() const;
   uint BytesWidth() const;

   unsigned char * Data(uint Row = 0);
   const unsigned char * Data(uint Row = 0) const;

   unsigned char * Data(uint x, uint y);
   const unsigned char * Data(uint x, uint y) const;

   void Create(const SImage& Img, uint Align = DefaultAlignement);
   void Create(uint W, uint H, uint C, SImage::EDataType T, uint Align = DefaultAlignement);

   template<class T>
   void Create(uint W, uint H, uint C = 1, uint Align = DefaultAlignement);

   virtual void Free();

   const SImage& ToSImage() const;

   void MakeBlack();

protected:
   unsigned char * m_Data;
   uint m_Offset;

private:
   // Not copyable
   CSimpleImage(const CSimpleImage&);
   CSimpleImage& operator = (const CSimpleImage&);
};

template<class T>
class CImage : public CSimpleImage
{
public:
   T& operator () (uint x, uint y)
   {
      T* Ptr = (T*) Data(y);
      return Ptr[x];
   }

   const T& operator () (uint x, uint y) const
   {
      const T* Ptr = (const T*) Data(y);
      return Ptr[x];
   }
};

class CImageROI : public CSimpleImage
{
public:
   CImageROI(CSimpleImage& Image, uint X, uint Y, uint W, uint H);
   virtual ~CImageROI() { m_Data = nullptr; }

private:
   // These should not be used on a ROI
   void Create(const SImage& Img, uint Align = DefaultAlignement);
   void Create(uint W, uint H, uint C, SImage::EDataType T, uint Align = DefaultAlignement);

   template<class T>
   void Create(uint W, uint H, uint C = 1, uint Align = DefaultAlignement);

   virtual void Free() { }
};

template<class T>
inline SImage::EDataType GetImageType();

#define IMG_TYPE(type, value) \
   template<> inline SImage::EDataType GetImageType<type>()\
   {\
      return SImage::value;\
   }

IMG_TYPE(unsigned char, U8)
IMG_TYPE(char, S8)
IMG_TYPE(unsigned short, U16)
IMG_TYPE(short, S16)
IMG_TYPE(uint, U32)
IMG_TYPE(int, S32)
IMG_TYPE(float, F32)
IMG_TYPE(double, F64)

#undef IMG_TYPE

template<class T>
inline void CSimpleImage::Create(uint W, uint H, uint C, uint Align)
{
   Create(W, H, C, GetImageType<T>(), Align);
}
