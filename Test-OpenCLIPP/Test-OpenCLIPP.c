////////////////////////////////////////////////////////////////////////////////
//! @file	: Test-OpenCLIPP.c
//! @date   : Jul 2013
//!
//! @brief  : Simple test program for C interface of OpenCLIPP library
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


#include <OpenCLIPP.h>

#include <malloc.h>
#include <stdio.h>

#include "png/lodepng.h"   // For reading&saving image files


int main(int argc, char * argv[])
{
   // Variables
   char DeviceName[100] = {'\0'};
   ocipContext Context = NULL;
   ocipError Error = CL_SUCCESS;

   SImage ImageInfo;
   void * SourceData = NULL;
   void * ResultData = NULL;
   ocipImage Source, Result;


   // Load source image
   unsigned Status = lodepng_decode24_file((unsigned char **) &SourceData, &ImageInfo.Width, &ImageInfo.Height, "lena.png");

   if (Status == 0)
   {
      // Load image from file succeded
      ImageInfo.Channels = 3;
      ImageInfo.Type = U8;
      ImageInfo.Step = ImageInfo.Width * ImageInfo.Channels;
   }
   else
   {
      // Load image from file failed
      printf("Unable to open file lena.png - using empty image\n");

      ImageInfo.Channels = 3;
      ImageInfo.Type = U8;
      ImageInfo.Width = 512;
      ImageInfo.Height = 512;
      ImageInfo.Step = ImageInfo.Width * ImageInfo.Channels;
      SourceData = malloc(ImageInfo.Step * ImageInfo.Height);
   }


   // Allocate destination image
   ResultData = malloc(ImageInfo.Step * ImageInfo.Height);


   // Tell where the .cl files are
   ocipSetCLFilesPath("D:/OpenCLIPP/cl files/");


   // Initialize
   Error = ocipInitialize(&Context, NULL, CL_DEVICE_TYPE_ALL);

   if (Error != CL_SUCCESS)
   {
      printf("Unable to initialize OpenCL\n");
      return Error;
   }

   // Display device name
   ocipGetDeviceName(DeviceName, 100);
   printf("Using OpenCL device : %s\n", DeviceName);

   // Allocate images on the device
   Error = ocipCreateImage(&Source, ImageInfo, SourceData, CL_MEM_READ_ONLY);
   Error = ocipCreateImage(&Result, ImageInfo, ResultData, CL_MEM_WRITE_ONLY);


   // Prepare to execute a filter
   Error = ocipPrepareFilters(Source);

   if (Error != CL_SUCCESS)
   {
      printf("Unable to prepare filters\n");
      return Error;
   }

   Error = ocipGaussianBlur(Source, Result, 6);


   // Read Result image from device
   Error = ocipReadImage(Result);


   // Save result to a file
   lodepng_encode24_file("result.png", (unsigned char*) ResultData, ImageInfo.Width, ImageInfo.Height);


   // Free images on the device
   Error = ocipReleaseImage(Source);
   Error = ocipReleaseImage(Result);


   // Free images
   free(SourceData);
   free(ResultData);


   // Uninitialize
   Error = ocipUninitialize(Context);

   if (Error != CL_SUCCESS)
      return 1;

   printf("Success\n");

   return 0;
}
