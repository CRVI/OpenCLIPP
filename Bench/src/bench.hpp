////////////////////////////////////////////////////////////////////////////////
//! @file	: bench.hpp
//! @date   : Jul 2013
//!
//! @brief  : Lists all benchmarks to run
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

#define USE_BUFFER true // Set to false to use Image mode

#include "benchBase.hpp"
#include "benchArithmetic.hpp"
#include "benchStatistics.hpp"
#include "benchMorphoBase.hpp"
#include "benchTopHat3x3.hpp"
#include "benchBlackHat3x3.hpp"
#include "benchGradient.hpp"
#include "benchMedian3x3.hpp"
#include "benchMedian5x5.hpp"
#include "benchConvert.hpp"
#include "benchScale.hpp"
#include "benchLut.hpp"
#include "benchLinearLut.hpp"
#include "benchTransfer.hpp"
#include "benchImageProximityBase.hpp"
#include "benchImageTransfer.hpp"
#include "benchIntegral.hpp"
#include "benchSqrIntegral.hpp"
#include "benchResize.hpp"
#include "benchFilters.hpp"
#include "benchThreshold.hpp"
#include "benchThresholdGTLT.hpp"
#include "benchCompare.hpp"
#include "benchThresholdImg.hpp"
#include "benchFFT.hpp"
#include "benchTransform.hpp"
#include "benchRotate.hpp"
#include "benchBlob.hpp"

void RunBench()
{
#ifdef FULL_TESTS

   // Unit test mode
#define B_NO_F(name) /*Not float*/\
   _BENCH2(name, unsigned char);\
   _BENCH2(name, unsigned short);

#define B(name) \
   B_NO_F(name);\
   _BENCH2(name, float);
   
   B(Add);
   //B(AddSquare);
   B(Sub);
   B(AbsDiff);
   B(Mul);
   B(Div);
   /*B(ImgMin);
   B(ImgMax);
   B(ImgMean);
   B(Combine);*/

   B(AddC);
   B(SubC);
   B(AbsDiffC);
#ifndef HAS_CV
   B(MulC);     // Fails with OpenCV OCL
   B(DivC);     // Fails with OpenCV OCL
#endif
   /*B(RevDivC);
   B(MinC);
   B(MaxC);
   B(MeanC);*/

   _BENCH2(Abs, float);

   _BENCH2(Exp, float); // OpenCV can do exp only on float images
   //B(Exp);
   B(Sqr);
   B(Sqrt);
   /*B(Sin);
   B(Cos);
   B(Log);
   B(Invert);*/

   B_NO_F(And);
   B_NO_F(Or);
   B_NO_F(Xor);
   B_NO_F(AndC);
   B_NO_F(OrC);
   B_NO_F(XorC);
   //B_NO_F(Not);

   B(Min);
   B(Max);
   B(Sum);
#ifndef HAS_CV
   B(Mean);  // mean crashes with OpenCV OCL
   B(Mean_StdDev);
#endif
   B(MinIndx);
   B(MaxIndx);

   B(Min4C);
   B(Max4C);
   B(Sum4C);
   B(Mean4C);

   Bench(ErodeBench);
   Bench(DilateBench);

   B(MirrorX);
   B(MirrorY);
   B(Flip);
#ifndef HAS_CV
   B(Transpose);  // OpenCV OCL does not accept U16 for transpose
#endif

   /*B(Rotate);      // Rotation works well but we seem to handle the image border in a slightly different way from IPP
   B(RotateLinear);*/

   // Reduce size
   B(Resize);  // Resize half
   B(ResizeLinear);  // Resize half with linear interpolation

   // Increase size
   B(ResizeBigger);
   B(ResizeBiggerLinear);

   // Filters
   Bench(Sobel3_1Bench);
   Bench(Sobel5_1Bench);
   Bench(Prewitt3_1Bench);
   Bench(Scharr3_1Bench);

   Bench(Gauss3_1Bench);
   Bench(Gauss5_1Bench);
   Bench(Sharpen3_1Bench);
   Bench(SobelVert3_1Bench);
   Bench(SobelHoriz3_1Bench);
   Bench(SobelVert5_1Bench);
   Bench(SobelHoriz5_1Bench);
   Bench(ScharrVert3_1Bench);
   Bench(ScharrHoriz3_1Bench);
   Bench(PrewittVert3_1Bench);
   Bench(PrewittHoriz3_1Bench);
   Bench(Laplace3_1Bench);
   Bench(Laplace5_1Bench);
   Bench(SobelCross3_1Bench);
   Bench(SobelCross5_1Bench);

   Bench(Sharpen3_4Bench);
   Bench(SobelVert3_4Bench);
   Bench(SobelHoriz3_4Bench);
#ifndef FULL_BENCH               // These work but they use too much RAM to run well on my machine with high resolution images - would require 12+GB of RAM for them
   Bench(PrewittVert3_4Bench);
   Bench(PrewittHoriz3_4Bench);
   Bench(Laplace3_4Bench);
   Bench(Gauss3_4Bench);
   Bench(Laplace5_4Bench);
   Bench(Gauss5_4Bench);
   Bench(Sobel3_4Bench);
   Bench(Prewitt3_4Bench);
#endif   // FULL_BENCH
   
   B(CompareCLT);
   B(CompareCLQ);
   B(CompareCEQ);
   B(CompareCGQ);
   B(CompareCGT);

   B(CompareLT);
   B(CompareLQ);
   B(CompareEQ);
   B(CompareGQ);
   B(CompareGT);

   
   B(ThresholdLT);
   B(ThresholdGT);
   //B(ThresholdLQ); // These 3 are not supported by IPP
   //B(ThresholdEQ);
   //B(ThresholdGQ);
   
   B(ThresholdGTLT);

   /* // This is not implemented in IPP
   B(Threshold_ImgLT);
   B(Threshold_ImgLQ);
   B(Threshold_ImgEQ);
   B(Threshold_ImgGQ);
   B(Threshold_ImgGT);*/

   B(SqrDistance);

   if (ocipIsFFTAvailable())
   {
      Bench(FFTForwardBench);
      Bench(FFTBackwardBench);
   }

   Bench(IntegralBenchF32);
   //Bench(IntegralBenchF64);       // Not supported by IPP
   //Bench(SqrIntegralBenchF32);    // Not supported by IPP
   if (USE_BUFFER)                  // F64 is only supported using buffer
   {
      Bench(SqrIntegralBenchF64);
   }

   Bench(BlobBench);

#else // FULL_TESTS
   // Benchmark mode
   Bench(TransferBench);
   Bench(ImageTransferBench);

   Bench(AbsDiffCBenchU8);
   Bench(AbsDiffBenchU8);
   Bench(MaxBenchU8);
   Bench(MeanBenchU8);

   Bench(AbsDiffCBenchU16);
   Bench(AbsDiffBenchU16);
   Bench(MaxBenchU16);
   Bench(MeanBenchU16);

   Bench(AbsDiffCBenchF32);
   Bench(AbsDiffBenchF32);
   Bench(MaxBenchF32);
   Bench(MeanBenchF32);

#endif // FULL_TESTS

   Bench(TopHatBench);
   Bench(BlackHatBench);
   Bench(GradientBench);
   Bench(Median3x3Bench);
   Bench(Median5x5Bench);

   Bench(ConvertBenchU8);
   Bench(ConvertBenchU16);
   Bench(ConvertBenchS16);
   Bench(ConvertBenchS32);
   Bench(ConvertBenchF32);
   Bench(ConvertBenchFU8);

   Bench(ScaleBenchU8);

   Bench(LutBenchU8);
   Bench(LinearLutBenchF32);
}
