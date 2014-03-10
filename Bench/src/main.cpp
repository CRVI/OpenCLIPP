////////////////////////////////////////////////////////////////////////////////
//! @file	: main.cpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark program for image processing libraries
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

#include "config.h"
//-------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "CImage.h"
#include "Timer.h"

//#define HAS_IPP   // Intel IPP library
//#define HAS_NPP   // nVidia NPP library
//#define HAS_CV    // OpenCV OCL library
//#define HAS_CUDA  // Custom CUDA implementation

#if defined(FULL_TESTS) && !defined(HAS_IPP)
#error Need IPP to do the full tests - IPP is used as reference
#endif


#include "preprocessor.h"

#include "Libraries.h"

void FillRandomImg(CSimpleImage& Image, int ImageNo = 0);

//-------------------------------------------------------------------------------------------------
#ifdef FULL_BENCH
#ifdef FULL_TESTS
 const uint BENCH_ITERATIONS = 1;   // FULL_TESTS checks correctness, not speed
#else
 const uint BENCH_ITERATIONS = 30;  // Do many runs of each and average the time
#endif
 const uint BENCH_COUNT = 25;
 const char* BENCH_NAME[BENCH_COUNT] = { "CGA      ", "EGA      ", "VGA      ", "SVGA     ", "XGA      ", "SXGA     ", "UXGA     ", "WUXGA    ", "QXGA     ", "QSXGA    ", "QUXGA    ", "WQUXGA   ", "HXGA     ", "HSXGA    ", "HUXGA    ", "WHUXGA   ", "16x16    ", "32x32    ", "64x64    ", "128x128  ", "256x256  ", "512x512  ", "1024x1024", "2048x2048", "4096x4096" };
 uint BENCH_WIDTH[BENCH_COUNT]   = { 320, 640, 640, 800, 1024, 1280, 1600, 1920, 2048, 2560, 3200, 3840, 4096, 5120, 6400, 7680, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
 uint BENCH_HEIGHT[BENCH_COUNT]  = { 200, 350, 480, 600,  768, 1024, 1200, 1200, 1536, 2048, 2400, 2400, 3072, 4096, 4800, 4800, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
#else
 const uint BENCH_ITERATIONS = 1;
 const uint BENCH_COUNT = 1;
 const char* BENCH_NAME[BENCH_COUNT] = {"1024x1024"};
 uint BENCH_WIDTH[BENCH_COUNT]   = { 1024 };
 uint BENCH_HEIGHT[BENCH_COUNT]  = { 1024 };
#endif
//-------------------------------------------------------------------------------------------------
#define Bench(BenchableType) _BENCH(BenchableType, #BenchableType)
#define _BENCH2(bench_type, data_type) _BENCH(CONCATENATE(bench_type, Bench)<data_type>, STR(bench_type-data_type))
#define _BENCH2T(bench_type, data_type1, data_type2) \
   _BENCH(__ID(CONCATENATE(bench_type, Bench)<data_type1, data_type2>), STR(bench_type-data_type1))
#define _BENCH(type, name)    \
{                             \
   type instance;             \
   runbench(name, instance);  \
}
void Check(bool Result, const char * Lib, std::string& String)
{
   if (!IPP_AVAILABLE)
   {
      String = "No IPP - can't compare";
      return;
   }

   if (Result)
      return;

   if (String == "Success")
      String = "Fail ";
   else
      String += "&";

   String += Lib;
}

struct Libraries
{
   enum ELibs
   {
      IPP,
      CL,
      NPP,
      CV,
      CUDA,
      NbLibs,
   };

   ELibs ID;
   double Time;
   bool Available;
   bool Success;

   const char * Name()
   {
      switch (ID)
      {
      case CL:
         return "CL";
      case IPP:
         return "IPP";
      case NPP:
         return "NPP";
      case CV:
         return "CV";
      case CUDA:
         return "CUDA";
      default:
         return "";
      }

   }

};
//-------------------------------------------------------------------------------------------------
template<class BenchableType>
void runbench(const char* szBenchname, BenchableType& cBenchable)
{
   Libraries Libs[Libraries::NbLibs];

   Libs[Libraries::CL].Available = true && cBenchable.HasCLTest();
   Libs[Libraries::NPP].Available = NPP_AVAILABLE && cBenchable.HasNPPTest();
   Libs[Libraries::IPP].Available = IPP_AVAILABLE;
   Libs[Libraries::CUDA].Available = CUDA_AVAILABLE && cBenchable.HasCUDATest();
   Libs[Libraries::CV].Available = CV_AVAILABLE && cBenchable.HasCVTest();

   for (int i = 0; i < Libraries::NbLibs; i++)
      Libs[i].ID = (Libraries::ELibs) i;

#ifndef FULL_TESTS
   printf("%s", szBenchname);
   for (int i = 0; i < Libraries::NbLibs; i++)
      if (Libs[i].Available)
         printf("\t%s\t", Libs[i].Name());

   printf("\tComparison\n");
#endif   // FULL_TESTS

   for(uint i = 0; i < BENCH_COUNT; i++)
   {
      CTimer Timer;

      //Alloc the images and prepare the images
      cBenchable.Create(BENCH_WIDTH[i], BENCH_HEIGHT[i]);

      if (Libs[Libraries::IPP].Available)
      {
         //Warm up the cache
         cBenchable.RunIPP();

         //Run a few times to bench
         Timer.Start();
         for(uint j = 0; j < BENCH_ITERATIONS; j++)
         {
            cBenchable.RunIPP();
         }
         Libs[Libraries::IPP].Time = Timer.Readms() / BENCH_ITERATIONS;

         Libs[Libraries::IPP].Success = true;    // IPP is the reference - always consider it successful
      }

      std::string Result = "Success";

      if (Libs[Libraries::CL].Available)
      {
         //Build the program
         cBenchable.RunCL();

         //Warm the cache
         cBenchable.RunCL();

         ocipFinish();

         //Run a few times to bench
         Timer.Start();
         for(uint j = 0; j < BENCH_ITERATIONS; j++)
         {
            cBenchable.RunCL();
         }

         ocipFinish();

         Libs[Libraries::CL].Time = Timer.Readms() / BENCH_ITERATIONS;

         //validate
         bool Success = cBenchable.CompareCL(&cBenchable);

         Check(Success, "CL", Result);

         Libs[Libraries::CL].Success = Success;
      }

      if (Libs[Libraries::NPP].Available)
      {
         //Warm the cache
         cBenchable.RunNPP();

         NPP_CODE(cudaDeviceSynchronize();)

         //Run a few times to bench
         Timer.Start();
         for(uint j = 0; j < BENCH_ITERATIONS; j++)
         {
            cBenchable.RunNPP();
         }

         NPP_CODE(cudaDeviceSynchronize();)

         Libs[Libraries::NPP].Time = Timer.Readms() / BENCH_ITERATIONS;

         bool Success = cBenchable.CompareNPP(&cBenchable);

         Check(Success, "NPP", Result);

         Libs[Libraries::NPP].Success = Success;
      }

      if (Libs[Libraries::CV].Available)
      {
         //Warm the cache
         cBenchable.RunCV();

         CV_CODE(finish();)

         //Run a few times to bench
         Timer.Start();
         for(uint j = 0; j < BENCH_ITERATIONS; j++)
         {
            cBenchable.RunCV();
         }

         CV_CODE(finish();)

         Libs[Libraries::CV].Time = Timer.Readms() / BENCH_ITERATIONS;

         bool Success = cBenchable.CompareCV(&cBenchable);

         Check(Success, "CV", Result);

         Libs[Libraries::CV].Success = Success;
      }

      if (Libs[Libraries::CUDA].Available)
      {
         //Warm the cache
         cBenchable.RunCUDA();

         CUDA_CODE(CUDA_WAIT)

         //Run a few times to bench
         Timer.Start();
         for(uint j = 0; j < BENCH_ITERATIONS; j++)
         {
            cBenchable.RunCUDA();
         }

         CUDA_CODE(CUDA_WAIT)

         Libs[Libraries::CUDA].Time = Timer.Readms() / BENCH_ITERATIONS;

         bool Success = cBenchable.CompareCUDA(&cBenchable);

         Check(Success, "CUDA", Result);

         Libs[Libraries::CUDA].Success = Success;
      }

#ifdef FULL_TESTS
      if (Libs[Libraries::IPP].Available)
         for (int j = 0; j < Libraries::NbLibs; j++)
            if (Libs[j].Available && !Libs[j].Success)
            {
               printf("Failed test : %s - %s - %s\n", szBenchname, BENCH_NAME[i], Libs[j].Name());
            }
#else
      printf("%s", BENCH_NAME[i]);

      for (int i = 0; i < Libraries::NbLibs; i++)
         if (Libs[i].Available)
            printf("\t%10.7f", Libs[i].Time);

      printf("\t[%s]\n", Result.c_str());
#endif   // FULL_TESTS

      //Cleanup
      cBenchable.Free();
   }

#ifndef FULL_TESTS
   printf("\n");
#endif   // FULL_TESTS
}
//-------------------------------------------------------------------------------------------------
SSize GetBiggestImage()
{
   SSize Biggest(0, 0);
   for (uint i = 0; i < BENCH_COUNT; i++)
   {
      if (BENCH_WIDTH[i] > Biggest.Width)
         Biggest.Width = BENCH_WIDTH[i];

      if (BENCH_HEIGHT[i] > Biggest.Height)
         Biggest.Height = BENCH_HEIGHT[i];
   }

   return Biggest;
}
//-------------------------------------------------------------------------------------------------
template<class T>
inline void BinarizeImg(CImage<T>& Image)
{
   for (uint y = 0; y < Image.Height; y++)
      for (uint x = 0; x < Image.Width; x++)
      {
         T& V = Image(x, y);
         if (V > 63)
            V = 255;
         else
            V = 0;
      }

}
//-------------------------------------------------------------------------------------------------
#ifdef HAS_IPP
#include "Compare.h"
#else
template<class T>
static inline bool CompareImages(const CSimpleImage&, const CSimpleImage&, const CSimpleImage&, const T&)
{
   return false;
}
#endif   // HAS_IPP
//-------------------------------------------------------------------------------------------------
void RunBench();
//-------------------------------------------------------------------------------------------------
int main()
{
   ocipContext CLContext;

   ocipInitialize(&CLContext, "", CL_DEVICE_TYPE_ALL);
   //ocipInitialize(&CLContext, "Intel", CL_DEVICE_TYPE_CPU);   // Use this to use the Intel OpenCL debugger

   ocipSetCLFilesPath("D:/OpenCLIPP/cl-files/");

#ifdef FULL_TESTS
   printf("Running unit tests using randomly generated images\n");
   printf("All primitives having an IPP equivalent and a test class are run and compared with IPP\n");
   printf("Only failing tests will be listed below\n");
#else    // FULL_TESTS
   printf("Starting Bench using randomly generated images\n");
   printf("Each test is run %d times, result is average time per run in ms\n", BENCH_ITERATIONS);
#endif   // FULL_TESTS

   char CLDeviceName[100];
   ocipGetDeviceName(CLDeviceName, 100);

   printf("Bench using: %s for OpenCL\n", CLDeviceName);

   RunBench();

   #ifdef WAITFORKEY_AT_END
      getchar();
   #endif

   ocipUninitialize(CLContext);

   return 0;
}
//-------------------------------------------------------------------------------------------------
#include "bench.hpp"
//-------------------------------------------------------------------------------------------------
