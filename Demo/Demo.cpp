////////////////////////////////////////////////////////////////////////////////
//! @file	: Demo.cpp
//! @date   : Jul 2013
//!
//! @brief  : Simple demo program for OpenCLIPP library
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

#include <OpenCLIPP.hpp>

#include "png/lodepng.h"


using namespace std;
using namespace OpenCLIPP;

uint CalculateStep(const SImage& Img);


int main(int /*argc*/, char ** /*argv*/)
{
   // Load image from file
   string FileName = "lena.png";

   SImage ImageInfo;
   vector<unsigned char> SourceData, ResultData;

   unsigned Status = lodepng::decode(SourceData, ImageInfo.Width, ImageInfo.Height, FileName, LCT_RGB);

   if (Status == 0)
   {
      // Load image from file succeded
      ImageInfo.Channels = 3;
      ImageInfo.Type = SImage::U8;
      ImageInfo.Step = CalculateStep(ImageInfo);
   }
   else
   {
      // Load image from file failed
      printf("Unable to open file lena.png - using empty image\n");

      ImageInfo.Channels = 3;
      ImageInfo.Type = SImage::U8;
      ImageInfo.Width = 512;
      ImageInfo.Height = 512;
      ImageInfo.Step = CalculateStep(ImageInfo);
      SourceData.resize(ImageInfo.Height * ImageInfo.Step);
   }

   ResultData.resize(SourceData.size());


   // Initialize OpenCL
   COpenCL CL;
   CL.SetClFilesPath("D:/OpenCLIPP/cl-files/");
   Filters Filters(CL);

   // Display device name
   string Name = CL.GetDeviceName();
   printf("Using device : %s\n", Name.c_str());

   // Create images in OpenCL device
   Image SourceImage(CL, ImageInfo, SourceData.data());
   Image ResultImage(CL, ImageInfo, ResultData.data());

   SourceImage.Send();   // Optional : Sends image to device memory - would be done automatically


   // Execute filter
   Filters.Sobel(SourceImage, ResultImage);


   // Read image to main memory
   ResultImage.Read(true);


   // Save image to new file
   string ResultName = "result.png";
   lodepng::encode(ResultName, ResultData, ImageInfo.Width, ImageInfo.Height, LCT_RGB);

   printf("Success\n");

   return 0;
}

// Helper
uint CalculateStep(const SImage& Img)
{
   uint Depth = 4;
   if (Img.Type < SImage::U16)
      Depth = 1;
   else if (Img.Type < SImage::U32)
      Depth = 2;

   return Img.Width * Img.Channels * Depth;
}
