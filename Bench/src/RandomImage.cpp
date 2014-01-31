////////////////////////////////////////////////////////////////////////////////
//! @file	: RandomImage.cpp
//! @date   : Jul 2013
//!
//! @brief  : Implementation of FillRandomImg()
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

#include <random>
#include <memory.h>
#include "CImage.h"

SSize GetBiggestImage();

void FillRandomImg(CSimpleImage& Image, int ImageNo)
{
   static bool Initialized = false;
   const static int NbImages = 2;
   static CSimpleImage Images[NbImages];
   static CImage<float> FloatImages[NbImages];

   //assert(ImageNo >= 0 && ImageNo < NbImages);

   if (!Initialized)
   {
      // Random image generation is time consuming
      // So we do it only once

      std::mt19937 Rand;   // Mersenne twister pseudo random number generator

      // Uniform distribution that can set all but the leftmost bit
      std::uniform_int_distribution<unsigned char> Dist(0, 0xFF);

      SSize Big = GetBiggestImage();

      for (int i = 0; i < NbImages; i++)
      {
         Images[i].Create(Big.Width, Big.Height, 1, SImage::S32);
         uint uByteWidth = Images[i].BytesWidth();
         for (uint y = 0; y < Images[i].Height; y++)
            for (uint b = 0; b < uByteWidth; b++)
               Images[i].Data(y)[b] = Dist(Rand);
      }


      std::normal_distribution<float> FloatDist(0, 1);

      for (int i = 0; i < NbImages; i++)
      {
         FloatImages[i].Create(Big.Width, Big.Height, 1, SImage::F32);
         for (uint y = 0; y < FloatImages[i].Height; y++)
            for (uint x = 0; x < FloatImages[i].Width; x++)
               FloatImages[i](x, y) = FloatDist(Rand);
      }

      Initialized = true;
   }

   uint uByteWidth = Image.BytesWidth();
   
   if (Image.Type == Image.F32)
   {
      for (uint y = 0; y < Image.Height; y++)
         memcpy(Image.Data(y), FloatImages[ImageNo].Data(y), uByteWidth);

      return;
   }

   for (uint y = 0; y < Image.Height; y++)
      memcpy(Image.Data(y), Images[ImageNo].Data(y), uByteWidth);
}
