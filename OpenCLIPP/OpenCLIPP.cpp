////////////////////////////////////////////////////////////////////////////////
//! @file	: OpenCLIPP.cpp 
//! @date   : Jul 2013
//!
//! @brief  : Glue code between C interface and the rest of the library
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


/// About this file
/// This file implements the functions in OpenCLIPP.h by calling the equivalent
/// method in the C++ part of the library.
/// In this file, preprocessor macros are used to reduce the amount of repeated code.


// This needs to be defined before including <cl/cl.h> because we use OpenCL API version 1.1
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS


#include "OpenCLIPP.h"
#include "OpenCLIPP.hpp"

#include <map>

using namespace std;
using namespace OpenCLIPP;


struct SProgramList
{
   SProgramList(COpenCL& CL)
   :  arithmetic(CL),
      arithmeticVector(CL),
      conversions(CL),
      conversionsBuffer(CL),
      filters(CL),
      filtersVector(CL),
      histogram(CL),
      histogramBuffer(CL),
      logic(CL),
      logicVector(CL),
      lut(CL),
      lutVector(CL),
      morphology(CL),
      morphologyBuffer(CL),
      imageProximity(CL),
      imageProximityBuffer(CL),
      transform(CL),
      transformBuffer(CL),
      thresholding(CL),
      thresholdingVector(CL)
   { }

   Arithmetic arithmetic;
   ArithmeticVector arithmeticVector;
   Conversions conversions;
   ConversionsBuffer conversionsBuffer;
   Filters filters;
   FiltersVector filtersVector;
   Histogram histogram;
   HistogramBuffer histogramBuffer;
   Logic logic;
   LogicVector logicVector;
   Lut lut;
   LutVector lutVector;
   Morphology morphology;
   MorphologyBuffer morphologyBuffer;
   ImageProximity imageProximity;
   ImageProximityBuffer imageProximityBuffer;
   Transform transform;
   TransformBuffer transformBuffer;
   Thresholding thresholding;
   ThresholdingVector thresholdingVector;
};

// List of program for each context
map<COpenCL*, shared_ptr<SProgramList>> g_ProgramList;

COpenCL * g_CurrentContext = nullptr;

// Retreived the program list for the current context
SProgramList& GetList();


// Catches exceptions and returns proper error code
#define H(code) try { code ; } catch (cl::Error e) { return e.err(); } return CL_SUCCESS;

#define Img(i) (*((IImage*) i))
#define Buf(b) (*((ImageBuffer*) b))


// Outputs a concatenation of the two given arguments : CONCATENATE(ab, cd) -> abcd
#define CONCATENATE(a, b) _CONCATENATE(a, b)
#define _CONCATENATE(a, b) a ## b


// Initialization
ocipError ocip_API ocipInitialize(ocipContext * ContextPtr, const char * PreferredPlatform, cl_device_type deviceType)
{
   H(
      COpenCL * CL = new COpenCL(PreferredPlatform, deviceType);

      g_ProgramList[CL] = make_shared<SProgramList>(*CL);

      g_CurrentContext = CL;

      *ContextPtr = (ocipContext) CL;
      )
}

ocipError ocip_API ocipUninitialize(ocipContext Context)
{
   COpenCL * CL = (COpenCL *) Context;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

   if (CL == g_CurrentContext)
      g_CurrentContext = nullptr;

   H( g_ProgramList[CL].reset();
      delete CL; )
}

ocipError ocip_API ocipChangeContext(ocipContext Context)
{
   COpenCL * CL = (COpenCL *) Context;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

   if (g_ProgramList[CL] == nullptr)
      return CL_INVALID_CONTEXT;

   g_CurrentContext = CL;

   return CL_SUCCESS;
}

void ocip_API ocipSetCLFilesPath(const char * Path)
{
   COpenCL::SetClFilesPath(Path);
}

ocip_API const char * ocipGetErrorName(ocipError Error)
{
   return COpenCL::ErrorName(Error);
}

ocipError ocip_API ocipGetDeviceName(char * Name, uint BufferLength)
{
   COpenCL * CL = g_CurrentContext;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

#ifdef _MSC_VER
#pragma warning ( disable : 4996 )  // Potentially unsafe use of string::copy()
#endif   // _MSC_VER

   H(
      std::string DeviceName = CL->GetDeviceName();

      if (DeviceName.length() > BufferLength - 1)
         return CL_OUT_OF_RESOURCES;

      DeviceName.copy(Name, BufferLength, 0);
      Name[DeviceName.length()] = '\0';
      )
}

ocipError ocip_API ocipFinish()
{
   COpenCL * CL = g_CurrentContext;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

   H( CL->GetQueue().finish() );
}


ocipError ocip_API ocipCreateImageBuffer(ocipBuffer * BufferPtr, SImage image, void * ImageData, cl_mem_flags flags)
{
   COpenCL * CL = g_CurrentContext;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

   H( *BufferPtr = (ocipBuffer) new ImageBuffer(*CL, image, ImageData, flags) )
}

ocipError ocip_API ocipSendImageBuffer(ocipBuffer Buffer)
{
   IBuffer * Ptr = (IBuffer *) Buffer;
   ImageBuffer * Buf = dynamic_cast<ImageBuffer *>(Ptr);
   if (Buf == nullptr)
      return CL_INVALID_MEM_OBJECT;

   H( Buf->Send() )
}

ocipError ocip_API ocipReadImageBuffer(ocipBuffer Buffer)
{
   IBuffer * Ptr = (IBuffer *) Buffer;
   ImageBuffer * Buf = dynamic_cast<ImageBuffer *>(Ptr);
   if (Buf == nullptr)
      return CL_INVALID_MEM_OBJECT;

   H( Buf->Read(true) )
}

ocipError ocip_API ocipReleaseImageBuffer(ocipBuffer Buffer)
{
   Memory * Mem = (Memory *) Buffer;
   H( delete Mem )
}

ocipError ocip_API ocipCreateImage(ocipImage * ImagePtr, SImage image, void * ImageData, cl_mem_flags flags)
{
   COpenCL * CL = g_CurrentContext;

   if (CL == nullptr)
      return CL_INVALID_CONTEXT;

   H( if (image.Channels == 3)
         *ImagePtr = (ocipImage) new ColorImage(*CL, image, ImageData);
      else
         *ImagePtr = (ocipImage) new Image(*CL, image, ImageData, flags);
       )
}

ocipError ocip_API ocipSendImage(ocipImage image)
{
   IImage * Ptr = (IImage *) image;
   Image * Img = dynamic_cast<Image *>(Ptr);
   if (Img != nullptr)
   {
      H( Img->Send() )
   }

   ColorImage * ColorImg = dynamic_cast<ColorImage *>(Ptr);
   if (ColorImg == nullptr)
      return CL_INVALID_MEM_OBJECT;

   H(ColorImg->Send());
}

ocipError ocip_API ocipReadImage(ocipImage image)
{
   IImage * Ptr = (IImage *) image;
   Image * Img = dynamic_cast<Image *>(Ptr);
   if (Img != nullptr)
   {
      H( Img->Read(true) );
   }

   ColorImage * ColorImg = dynamic_cast<ColorImage *>(Ptr);
   if (ColorImg == nullptr)
      return CL_INVALID_MEM_OBJECT;

   H(ColorImg->Read(true));
}

ocipError ocip_API ocipReleaseImage(ocipImage image)
{
   Memory * Mem = (Memory *) image;
   H( delete Mem )
}


ocipError ocip_API ocipReleaseProgram(ocipProgram Program)
{
   MultiProgram * Pr = (MultiProgram *) Program;
   H( delete Pr )
}


// Macros for less code repetition

// For implementing standard ocipPrepare* functions
#define PREPARE(fun, Class) \
ocipError ocip_API fun(IMAGE_ARG Image)\
{\
   H( GetList().Class.PrepareFor(CONV(Image)) );\
}


// For implementing ocipPrepare* functions that return a program
#define PREPARE2(name, Class) \
ocipError ocip_API name(ocipProgram * ProgramPtr, IMAGE_ARG Image)\
{\
   H(\
      if (g_CurrentContext == nullptr)\
         return CL_INVALID_CONTEXT;\
      Class * Ptr = new Class(*g_CurrentContext);\
      *ProgramPtr = (ocipProgram) Ptr;\
      if (Image != nullptr)\
         Ptr->PrepareFor(CONV(Image));\
   )\
}


// Macros to implement most primitives
#define BINARY_OP(fun, method) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source1, IMAGE_ARG Source2, IMAGE_ARG Dest)\
{\
   H( CLASS.method(CONV(Source1), CONV(Source2), CONV(Dest)) )\
}

#define CONSTANT_OP(fun, method, type) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, IMAGE_ARG Dest, type value)\
{\
   H( CLASS.method(CONV(Source), CONV(Dest), value) )\
}

#define UNARY_OP(fun, method) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, IMAGE_ARG Dest)\
{\
   H( CLASS.method(CONV(Source), CONV(Dest)) )\
}

#define REDUCE_OP(fun, method, type) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, type Result)\
{\
   H( CLASS.method(CONV(Source), Result) )\
}

#define REDUCE_RETURN_OP(fun, method, type) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, type * Result)\
{\
   H( *Result = CLASS.method(CONV(Source)) )\
}

#define REDUCE_INDEX_OP(fun, method, type) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, type * Result, int * IndexX, int * IndexY)\
{\
   H( *Result = CLASS.method(CONV(Source), *IndexX, *IndexY) )\
}

#define REDUCE_ARG_OP(fun, method, type) \
ocipError ocip_API fun(PROGRAM_ARG IMAGE_ARG Source, type * Result)\
{\
   H( CLASS.method(CONV(Source), Result) )\
}


// Image based operations
#define CONV Img

#define PROGRAM_ARG 

#define IMAGE_ARG ocipImage

PREPARE(ocipPrepareImageArithmetic, arithmetic)
PREPARE(ocipPrepareImageLogic, logic)
PREPARE(ocipPrepareImageLUT, lut)
PREPARE(ocipPrepareMorphology, morphology)
PREPARE(ocipPrepareTransform, transform)
PREPARE(ocipPrepareConversion, conversions)
PREPARE(ocipPrepareThresholding, thresholding)
PREPARE(ocipPrepareFilters, filters)
PREPARE(ocipPrepareHistogram, histogram)
PREPARE(ocipPrepareImageProximity, imageProximity)

PREPARE2(ocipPrepareStatistics, Statistics)
PREPARE2(ocipPrepareIntegral, Integral)


#define CLASS GetList().arithmetic

BINARY_OP(ocipAdd, Add)
BINARY_OP(ocipAddSquare, AddSquare)
BINARY_OP(ocipSub, Sub)
BINARY_OP(ocipAbsDiff, AbsDiff)
BINARY_OP(ocipMul, Mul)
BINARY_OP(ocipDiv, Div)
BINARY_OP(ocipImgMin, Min)
BINARY_OP(ocipImgMax, Max)
BINARY_OP(ocipImgMean, Mean)
BINARY_OP(ocipCombine, Combine)

CONSTANT_OP(ocipAddC, Add, float)
CONSTANT_OP(ocipSubC, Sub, float)
CONSTANT_OP(ocipAbsDiffC, AbsDiff, float)
CONSTANT_OP(ocipMulC, Mul, float)
CONSTANT_OP(ocipDivC, Div, float)
CONSTANT_OP(ocipRevDivC, RevDiv, float)
CONSTANT_OP(ocipMinC, Min, float)
CONSTANT_OP(ocipMaxC, Max, float)
CONSTANT_OP(ocipMeanC, Mean, float)

UNARY_OP(ocipAbs, Abs)
UNARY_OP(ocipInvert, Invert)
UNARY_OP(ocipExp, Exp)
UNARY_OP(ocipLog, Log)
UNARY_OP(ocipSqr, Sqr)
UNARY_OP(ocipSqrt, Sqrt)
UNARY_OP(ocipSin, Sin)
UNARY_OP(ocipCos, Cos)


#undef CLASS
#define CLASS GetList().logic

BINARY_OP(ocipAnd, And)
BINARY_OP(ocipOr, Or)
BINARY_OP(ocipXor, Xor)

CONSTANT_OP(ocipAndC, And, uint)
CONSTANT_OP(ocipOrC, Or, uint)
CONSTANT_OP(ocipXorC, Xor, uint)

UNARY_OP(ocipNot, Not)


#undef CLASS
#define CLASS GetList().lut

ocipError ocip_API ocipLut(ocipImage Source, ocipImage Dest, uint * levels, uint * values, int NbValues)
{
   H( CLASS.LUT(Img(Source), Img(Dest), levels, values, NbValues) )
}

ocipError ocip_API ocipLutLinear(ocipImage Source, ocipImage Dest, float * levels, float * values, int NbValues)
{
   H( CLASS.LUTLinear(Img(Source), Img(Dest), levels, values, NbValues) )
}

ocipError ocip_API ocipLutScale(ocipImage Source, ocipImage Dest, float SrcMin, float SrcMax, float DstMin, float DstMax)
{
   H( CLASS.Scale(Img(Source), Img(Dest), SrcMin, SrcMax, DstMin, DstMax) )
}


#undef CLASS
#define CLASS GetList().morphology

ocipError ocip_API ocipErode(ocipImage Source, ocipImage Dest, int Width)
{
   H( CLASS.Erode(Img(Source), Img(Dest), Width) )
}

ocipError ocip_API ocipDilate(ocipImage Source, ocipImage Dest, int Width)
{
   H( CLASS.Dilate(Img(Source), Img(Dest), Width) )
}

ocipError ocip_API ocipGradient(ocipImage Source, ocipImage Dest, ocipImage Temp, int Width)
{
   H( CLASS.Gradient(Img(Source), Img(Dest), Img(Temp), Width) )
}

#define MORPHO(fun, method) \
ocipError ocip_API fun(ocipImage Source, ocipImage Dest, ocipImage Temp, int Iterations, int Width)\
{\
   H( CLASS.method(Img(Source), Img(Dest), Img(Temp), Iterations, Width) )\
}

MORPHO(ocipErode2, Erode)
MORPHO(ocipDilate2, Dilate)
MORPHO(ocipOpen, Open)
MORPHO(ocipClose, Close)
MORPHO(ocipTopHat, TopHat)
MORPHO(ocipBlackHat, BlackHat)

#undef MORPHO

#undef CLASS
#define CLASS GetList().imageProximity

BINARY_OP(ocipSqrDistance_Norm, SqrDistance_Norm);
BINARY_OP(ocipSqrDistance, SqrDistance);
BINARY_OP(ocipAbsDistance, AbsDistance);
BINARY_OP(ocipCrossCorr, CrossCorr);
BINARY_OP(ocipCrossCorr_Norm, CrossCorr_Norm);

#undef CLASS
#define CLASS GetList().transform

UNARY_OP(ocipMirrorX, MirrorX)
UNARY_OP(ocipMirrorY, MirrorY)
UNARY_OP(ocipFlip, Flip)
UNARY_OP(ocipTranspose, Transpose)

ocipError ocip_API ocipRotate(ocipImage Source, ocipImage Dest, double Angle, double XShift, double YShift, enum ocipInterpolationType Interpolation)
{
   H( CLASS.Rotate(Img(Source), Img(Dest), Angle, XShift, YShift, Transform::EInterpolationType(Interpolation)) )
}

ocipError ocip_API ocipResize(ocipImage Source, ocipImage Dest, enum ocipInterpolationType Interpolation, ocipBool KeepRatio)
{
   H( CLASS.Resize(Img(Source), Img(Dest), Transform::EInterpolationType(Interpolation), KeepRatio != 0) )
}

ocipError ocip_API ocipSet(ocipImage Dest, float Value)
{
   H( CLASS.SetAll(Img(Dest), Value) )
}


#undef CLASS
#define CLASS GetList().conversions

UNARY_OP(ocipConvert, Convert)
UNARY_OP(ocipScale, Scale)
UNARY_OP(ocipCopy, Copy)
UNARY_OP(ocipToGray, ToGray)
UNARY_OP(ocipToColor, ToColor)

ocipError ocip_API ocipScale2(ocipImage Source, ocipImage Dest, int Offset, float Ratio)
{
   H( CLASS.Scale(Img(Source), Img(Dest), Offset, Ratio) )
}

ocipError ocip_API ocipCopy_B(ocipBuffer Source, ocipBuffer Dest)
{
   H( CLASS.Copy(Buf(Source), Buf(Dest)) )
}

ocipError ocip_API ocipToImage(ocipBuffer Source, ocipImage Dest)
{
   H( CLASS.Copy(Buf(Source), Img(Dest)) )
}

ocipError ocip_API ocipToBuffer(ocipImage Source, ocipBuffer Dest)
{
   H( CLASS.Copy(Img(Source), Buf(Dest)) )
}

ocipError ocip_API ocipSelectChannel(ocipImage Source, ocipImage Dest, int ChannelNo)
{
   H( CLASS.SelectChannel(Img(Source), Img(Dest), ChannelNo) )
}


#undef CLASS
#define CLASS GetList().thresholding

ocipError ocip_API ocipThresholdGTLT(ocipImage Source, ocipImage Dest, float threshLT, float valueLower, float threshGT, float valueHigher)
{
   H( CLASS.ThresholdGTLT(Img(Source), Img(Dest), threshLT, valueLower, threshGT, valueHigher) )
}

ocipError ocip_API ocipThreshold(ocipImage Source, ocipImage Dest, float Thresh, float value, ECompareOperation Op)
{
   H( CLASS.Threshold(Img(Source), Img(Dest), Thresh, value, (Thresholding::ECompareOperation) Op) )
}

ocipError ocip_API ocipThreshold_Img(ocipImage Source1, ocipImage Source2, ocipImage Dest, ECompareOperation Op)
{
   H( CLASS.Threshold(Img(Source1), Img(Source2), Img(Dest), (Thresholding::ECompareOperation) Op) )
}

ocipError ocip_API ocipCompare(ocipImage Source1, ocipImage Source2, ocipImage Dest, ECompareOperation Op)
{
   H( CLASS.Compare(Img(Source1), Img(Source2), Img(Dest), (Thresholding::ECompareOperation) Op) )
}

ocipError ocip_API ocipCompareC(ocipImage Source, ocipImage Dest, float Value, ECompareOperation Op)
{
   H( CLASS.Compare(Img(Source), Img(Dest), Value, (Thresholding::ECompareOperation) Op) )
}


#undef CLASS
#define CLASS GetList().filters

CONSTANT_OP(ocipGaussianBlur, GaussianBlur, float)
CONSTANT_OP(ocipGauss, Gauss, int)
CONSTANT_OP(ocipSharpen, Sharpen, int)
CONSTANT_OP(ocipSmooth, Smooth, int)
CONSTANT_OP(ocipMedian, Median, int)
CONSTANT_OP(ocipSobelVert, SobelVert, int)
CONSTANT_OP(ocipSobelHoriz, SobelHoriz, int)
CONSTANT_OP(ocipSobelCross, SobelCross, int)
CONSTANT_OP(ocipSobel, Sobel, int)
CONSTANT_OP(ocipPrewittVert, PrewittVert, int)
CONSTANT_OP(ocipPrewittHoriz, PrewittHoriz, int)
CONSTANT_OP(ocipPrewitt, Prewitt, int)
CONSTANT_OP(ocipScharrVert, ScharrVert, int)
CONSTANT_OP(ocipScharrHoriz, ScharrHoriz, int)
CONSTANT_OP(ocipScharr, Scharr, int)
CONSTANT_OP(ocipHipass, Hipass, int)
CONSTANT_OP(ocipLaplace, Laplace, int)



#undef CLASS
#define CLASS GetList().histogram

REDUCE_OP(ociphistogram_1C, Histogram1C, uint *)
REDUCE_OP(ociphistogram_4C, Histogram4C, uint *)
REDUCE_RETURN_OP(ocipOtsuThreshold, OtsuThreshold, uint)


// Begin programs that can have more than 1 instances per context
#undef PROGRAM_ARG
#define PROGRAM_ARG ocipProgram Program, 

#undef CLASS
#define CLASS (*(Statistics*)Program)

REDUCE_ARG_OP(   ocipMin,           Min,           double)
REDUCE_ARG_OP(   ocipMax,           Max,           double)
REDUCE_ARG_OP(   ocipMinAbs,        MinAbs,        double)
REDUCE_ARG_OP(   ocipMaxAbs,        MaxAbs,        double)
REDUCE_ARG_OP(   ocipSum,           Sum,           double)
REDUCE_ARG_OP(   ocipSumSqr,        SumSqr,        double)
REDUCE_ARG_OP(   ocipMean,          Mean,          double)
REDUCE_ARG_OP(   ocipMeanSqr,       MeanSqr,       double)
REDUCE_ARG_OP(   ocipStdDev,        StdDev,        double)
REDUCE_RETURN_OP(ocipCountNonZero,  CountNonZero,  uint)
REDUCE_INDEX_OP( ocipMinIndx,       Min,           double)
REDUCE_INDEX_OP( ocipMaxIndx,       Max,           double)
REDUCE_INDEX_OP( ocipMinAbsIndx,    MinAbs,        double)
REDUCE_INDEX_OP( ocipMaxAbsIndx,    MaxAbs,        double)

ocipError ocip_API ocipMean_StdDev(PROGRAM_ARG IMAGE_ARG Source, double * Mean, double * StdDev)
{
   H( CLASS.StdDev(CONV(Source), StdDev, Mean) )
}


#undef CLASS
#define CLASS (*(Integral*)Program)

UNARY_OP(ocipIntegral, IntegralSum)
UNARY_OP(ocipSqrIntegral, SqrIntegral)



// Image buffer operations
#undef CONV
#define CONV Buf

#undef IMAGE_ARG
#define IMAGE_ARG ocipBuffer

#undef PROGRAM_ARG
#define PROGRAM_ARG 

PREPARE(ocipPrepareImageBufferConversion, conversionsBuffer)
PREPARE(ocipPrepareImageBufferArithmetic, arithmeticVector)
PREPARE(ocipPrepareImageBufferLogic, logicVector)
PREPARE(ocipPrepareImageBufferLUT, lutVector)
PREPARE(ocipPrepareImageBufferMorphology, morphologyBuffer)
PREPARE(ocipPrepareImageBufferFilters, morphologyBuffer)
PREPARE(ocipPrepareImageBufferThresholding, thresholdingVector)
PREPARE(ocipPrepareImageBufferProximity, imageProximityBuffer)

PREPARE2(ocipPrepareImageBufferStatistics, StatisticsVector)
PREPARE2(ocipPrepareImageBufferIntegral, IntegralBuffer)
PREPARE2(ocipPrepareBlob, Blob)


#undef CLASS
#define CLASS GetList().conversionsBuffer

UNARY_OP(ocipConvert_V, Convert)
UNARY_OP(ocipScale_V, Scale)
UNARY_OP(ocipCopy_V, Copy)
UNARY_OP(ocipToGray_V, ToGray)
UNARY_OP(ocipToColor_V, ToColor)

ocipError ocip_API ocipScale2_V(ocipBuffer Source, ocipBuffer Dest, int Offset, float Ratio)
{
   H( CLASS.Scale(Buf(Source), Buf(Dest), Offset, Ratio) )
}

ocipError ocip_API ocipSelectChannel_V(ocipBuffer Source, ocipBuffer Dest, int ChannelNo)
{
   H( CLASS.SelectChannel(Buf(Source), Buf(Dest), ChannelNo) )
}


#undef CLASS
#define CLASS GetList().arithmeticVector

BINARY_OP(ocipAdd_V, Add)
BINARY_OP(ocipAddSquare_V, AddSquare)
BINARY_OP(ocipSub_V, Sub)
BINARY_OP(ocipAbsDiff_V, AbsDiff)
BINARY_OP(ocipMul_V, Mul)
BINARY_OP(ocipDiv_V, Div)
BINARY_OP(ocipImgMin_V, Min)
BINARY_OP(ocipImgMax_V, Max)
BINARY_OP(ocipImgMean_V, Mean)
BINARY_OP(ocipCombine_V, Combine)

CONSTANT_OP(ocipAddC_V, Add, float)
CONSTANT_OP(ocipSubC_V, Sub, float)
CONSTANT_OP(ocipAbsDiffC_V, AbsDiff, float)
CONSTANT_OP(ocipMulC_V, Mul, float)
CONSTANT_OP(ocipDivC_V, Div, float)
CONSTANT_OP(ocipRevDivC_V, RevDiv, float)
CONSTANT_OP(ocipMinC_V, Min, float)
CONSTANT_OP(ocipMaxC_V, Max, float)
CONSTANT_OP(ocipMeanC_V, Mean, float)

UNARY_OP(ocipAbs_V, Abs)
UNARY_OP(ocipInvert_V, Invert)
UNARY_OP(ocipExp_V, Exp)
UNARY_OP(ocipLog_V, Log)
UNARY_OP(ocipSqr_V, Sqr)
UNARY_OP(ocipSqrt_V, Sqrt)
UNARY_OP(ocipSin_V, Sin)
UNARY_OP(ocipCos_V, Cos)


#undef CLASS
#define CLASS GetList().imageProximityBuffer

BINARY_OP(ocipSqrDistance_Norm_B, SqrDistance_Norm)
BINARY_OP(ocipSqrDistance_B, SqrDistance)
BINARY_OP(ocipAbsDistance_B, AbsDistance)
BINARY_OP(ocipCrossCorr_B, CrossCorr)
BINARY_OP(ocipCrossCorr_Norm_B, CrossCorr_Norm)


#undef CLASS
#define CLASS GetList().logicVector

BINARY_OP(ocipAnd_V, And)
BINARY_OP(ocipOr_V, Or)
BINARY_OP(ocipXor_V, Xor)

CONSTANT_OP(ocipAndC_V, And, uint)
CONSTANT_OP(ocipOrC_V, Or, uint)
CONSTANT_OP(ocipXorC_V, Xor, uint)

UNARY_OP(ocipNot_V, Not)


#undef CLASS
#define CLASS GetList().lutVector

ocipError ocip_API ocipLut_V(ocipBuffer Source, ocipBuffer Dest, uint * levels, uint * values, int NbValues)
{
   H( CLASS.LUT(Buf(Source), Buf(Dest), levels, values, NbValues) )
}

ocipError ocip_API ocipLutLinear_V(ocipBuffer Source, ocipBuffer Dest, float * levels, float * values, int NbValues)
{
   H( CLASS.LUTLinear(Buf(Source), Buf(Dest), levels, values, NbValues) )
}

ocipError ocip_API ocipBasicLut_V(ocipBuffer Source, ocipBuffer Dest, unsigned char * values)
{
   H( CLASS.BasicLut(Buf(Source), Buf(Dest), values) )
}

ocipError ocip_API ocipLutScale_V(ocipBuffer Source, ocipBuffer Dest, float SrcMin, float SrcMax, float DstMin, float DstMax)
{
   H( CLASS.Scale(Buf(Source), Buf(Dest), SrcMin, SrcMax, DstMin, DstMax) )
}



#undef CLASS
#define CLASS GetList().morphologyBuffer

ocipError ocip_API ocipErode_B(ocipBuffer Source, ocipBuffer Dest, int Width)
{
   H( CLASS.Erode(Buf(Source), Buf(Dest), Width) )
}

ocipError ocip_API ocipDilate_B(ocipBuffer Source, ocipBuffer Dest, int Width)
{
   H( CLASS.Dilate(Buf(Source), Buf(Dest), Width) )
}

ocipError ocip_API ocipGradient_B(ocipBuffer Source, ocipBuffer Dest, ocipBuffer Temp, int Width)
{
   H( CLASS.Gradient(Buf(Source), Buf(Dest), Buf(Temp), Width) )
}

#define MORPHO(fun, method) \
ocipError ocip_API CONCATENATE(fun, _B)(ocipBuffer Source, ocipBuffer Dest, ocipBuffer Temp, int Iterations, int Width)\
{\
   H( CLASS.method(Buf(Source), Buf(Dest), Buf(Temp), Iterations, Width) )\
}

MORPHO(ocipErode2, Erode)
MORPHO(ocipDilate2, Dilate)
MORPHO(ocipOpen, Open)
MORPHO(ocipClose, Close)
MORPHO(ocipTopHat, TopHat)
MORPHO(ocipBlackHat, BlackHat)



#undef CLASS
#define CLASS GetList().transformBuffer

UNARY_OP(ocipMirrorX_V, MirrorX)
UNARY_OP(ocipMirrorY_V, MirrorY)
UNARY_OP(ocipFlip_V, Flip)
UNARY_OP(ocipTranspose_V, Transpose)

ocipError ocip_API ocipRotate_V(ocipBuffer Source, ocipBuffer Dest, double Angle, double XShift, double YShift, enum ocipInterpolationType Interpolation)
{
   H( CLASS.Rotate(Buf(Source), Buf(Dest), Angle, XShift, YShift, TransformBuffer::EInterpolationType(Interpolation) ) )
}

ocipError ocip_API ocipResize_V(ocipBuffer Source, ocipBuffer Dest, enum ocipInterpolationType Interpolation, ocipBool KeepRatio)
{
   H( CLASS.Resize(Buf(Source), Buf(Dest), TransformBuffer::EInterpolationType(Interpolation), KeepRatio != 0) )
}

ocipError ocip_API ocipSet_V(ocipBuffer Dest, float Value)
{
   H( CLASS.SetAll(Buf(Dest), Value) )
}



#undef CLASS
#define CLASS GetList().filtersVector

CONSTANT_OP(ocipGaussianBlur_V, GaussianBlur, float)
CONSTANT_OP(ocipGauss_V, Gauss, int)
CONSTANT_OP(ocipSharpen_V, Sharpen, int)
CONSTANT_OP(ocipSmooth_V, Smooth, int)
CONSTANT_OP(ocipMedian_V, Median, int)
CONSTANT_OP(ocipSobelVert_V, SobelVert, int)
CONSTANT_OP(ocipSobelHoriz_V, SobelHoriz, int)
CONSTANT_OP(ocipSobelCross_V, SobelCross, int)
CONSTANT_OP(ocipSobel_V, Sobel, int)
CONSTANT_OP(ocipPrewittVert_V, PrewittVert, int)
CONSTANT_OP(ocipPrewittHoriz_V, PrewittHoriz, int)
CONSTANT_OP(ocipPrewitt_V, Prewitt, int)
CONSTANT_OP(ocipScharrVert_V, ScharrVert, int)
CONSTANT_OP(ocipScharrHoriz_V, ScharrHoriz, int)
CONSTANT_OP(ocipScharr_V, Scharr, int)
CONSTANT_OP(ocipHipass_V, Hipass, int)
CONSTANT_OP(ocipLaplace_V, Laplace, int)



#undef CLASS
#define CLASS GetList().histogramBuffer

REDUCE_OP(ocipHistogram_1C_B, Histogram1C, uint *)
REDUCE_OP(ocipHistogram_4C_B, Histogram4C, uint *)
REDUCE_RETURN_OP(ocipOtsuThreshold_B, OtsuThreshold, uint)




// Begin programs that can have more than 1 instances per context
#undef PROGRAM_ARG
#define PROGRAM_ARG ocipProgram Program, 

#undef CLASS
#define CLASS (*(StatisticsVector*)Program)

REDUCE_ARG_OP(   ocipMin_V,            Min,           double)
REDUCE_ARG_OP(   ocipMax_V,            Max,           double)
REDUCE_ARG_OP(   ocipMinAbs_V,         MinAbs,        double)
REDUCE_ARG_OP(   ocipMaxAbs_V,         MaxAbs,        double)
REDUCE_ARG_OP(   ocipSum_V,            Sum,           double)
REDUCE_ARG_OP(   ocipSumSqr_V,         SumSqr,        double)
REDUCE_ARG_OP(   ocipMean_V,           Mean,          double)
REDUCE_ARG_OP(   ocipMeanSqr_V,        MeanSqr,       double)
REDUCE_ARG_OP(   ocipStdDev_V,         StdDev,        double)
REDUCE_RETURN_OP(ocipCountNonZero_V,   CountNonZero,  uint)
REDUCE_INDEX_OP( ocipMinIndx_V,        Min,           double)
REDUCE_INDEX_OP( ocipMaxIndx_V,        Max,           double)
REDUCE_INDEX_OP( ocipMinAbsIndx_V,     MinAbs,        double)
REDUCE_INDEX_OP( ocipMaxAbsIndx_V,     MaxAbs,        double)

ocipError ocip_API ocipMean_StdDev_V(PROGRAM_ARG IMAGE_ARG Source, double * Mean, double * StdDev)
{
   H( CLASS.StdDev(CONV(Source), StdDev, Mean) )
}


#undef CLASS
#define CLASS (*(IntegralBuffer*)Program)

UNARY_OP(ocipIntegral_B, IntegralSum)
UNARY_OP(ocipSqrIntegral_B, SqrIntegral)

#undef CLASS
#define CLASS GetList().thresholdingVector


ocipError ocip_API ocipThresholdGTLT_V(ocipBuffer Source, ocipBuffer Dest, float threshLT, float valueLower, float threshGT, float valueHigher)
{
   H( CLASS.ThresholdGTLT(Buf(Source), Buf(Dest), threshLT, valueLower, threshGT, valueHigher) )
}

ocipError ocip_API ocipThreshold_V(ocipBuffer Source, ocipBuffer Dest, float Thresh, float value, ECompareOperation Op)
{
   H( CLASS.Threshold(Buf(Source), Buf(Dest), Thresh, value, (ThresholdingVector::ECompareOperation) Op) )
}

ocipError ocip_API ocipThreshold_Img_V(ocipBuffer Source1, ocipBuffer Source2, ocipBuffer Dest, ECompareOperation Op)
{
   H( CLASS.Threshold(Buf(Source1), Buf(Source2), Buf(Dest), (ThresholdingVector::ECompareOperation) Op) )
}

ocipError ocip_API ocipCompare_V(ocipBuffer Source1, ocipBuffer Source2, ocipBuffer Dest, ECompareOperation Op)
{
   H( CLASS.Compare(Buf(Source1), Buf(Source2), Buf(Dest), (ThresholdingVector::ECompareOperation) Op) )
}

ocipError ocip_API ocipCompareC_V(ocipBuffer Source, ocipBuffer Dest, float Value, ECompareOperation Op)
{
   H( CLASS.Compare(Buf(Source), Buf(Dest), Value, (ThresholdingVector::ECompareOperation) Op) )
}


#undef CLASS
#define CLASS (*(Blob*)Program)

ocipError ocip_API ocipComputeLabels(ocipProgram Program, ocipBuffer Source, ocipBuffer Labels, int ConnectType)
{
   H( CLASS.ComputeLabels(Buf(Source), Buf(Labels), ConnectType) )
}

ocipError ocip_API ocipRenameLabels(ocipProgram Program, ocipBuffer Labels)
{
   H( CLASS.RenameLabels(Buf(Labels)) )
}


#ifdef USE_CLFFT

#undef CLASS
#define CLASS (*(FFT*)Program)

ocipError ocip_API ocipPrepareFFT(ocipProgram * ProgramPtr, ocipBuffer RealImage, ocipBuffer ComplexImage)
{
   H(
      if (g_CurrentContext == nullptr)
         return CL_INVALID_CONTEXT;
      FFT * Ptr = new FFT(*g_CurrentContext);
      *ProgramPtr = (ocipProgram) Ptr;
      if (RealImage != nullptr && ComplexImage != nullptr)
         Ptr->PrepareFor(CONV(RealImage), CONV(ComplexImage));
   )
}

UNARY_OP(ocipFFTForward, Forward)
UNARY_OP(ocipFFTInverse, Inverse)

ocipBool  ocip_API ocipIsFFTAvailable()
{
   return 1;
}


#undef CLASS
#define CLASS (*(ImageProximityFFT*)Program)

ocipError ocip_API ocipPrepareImageProximityFFT(ocipProgram * ProgramPtr, ocipBuffer Image, ocipBuffer Template)
{
   H(
      if (g_CurrentContext == nullptr)
         return CL_INVALID_CONTEXT;
      ImageProximityFFT * Ptr = new ImageProximityFFT(*g_CurrentContext);
      *ProgramPtr = (ocipProgram) Ptr;
      if (Image != nullptr)
         Ptr->PrepareFor(CONV(Image), CONV(Template));
   )
}

ocipError ocip_API ocipSqrDistanceFFT(ocipProgram Program, ocipBuffer Source, ocipBuffer Template, ocipBuffer Dest)
{
   H( CLASS.SqrDistance(Buf(Source), Buf(Template), Buf(Dest)) )
}

ocipError ocip_API ocipSqrDistanceFFT_Norm(ocipProgram Program, ocipBuffer Source, ocipBuffer Template, ocipBuffer Dest)
{
   H( CLASS.SqrDistance_Norm(Buf(Source), Buf(Template), Buf(Dest)) )
}

ocipError ocip_API ocipCrossCorrFFT(ocipProgram Program, ocipBuffer Source, ocipBuffer Template, ocipBuffer Dest)
{
   H( CLASS.CrossCorr(Buf(Source), Buf(Template), Buf(Dest)) )
}

ocipError ocip_API ocipCrossCorrFFT_Norm(ocipProgram Program, ocipBuffer Source, ocipBuffer Template, ocipBuffer Dest)
{
   H( CLASS.CrossCorr_Norm(Buf(Source), Buf(Template), Buf(Dest)) )
}


#else // USE_CLFFT

#ifndef _MSC_VER  // Visual Studio does not support #warning
#warning "OpenCLIPP is not being built with clFFT - FFT operations will not be available"
#endif

// Library was not built with clFFT, FFT operations will not be supported
ocipBool  ocip_API ocipIsFFTAvailable()
{
   return 0;
}
ocipError ocip_API ocipPrepareFFT(ocipProgram *, ocipBuffer, ocipBuffer)
{
   return CL_INVALID_OPERATION;
}
ocipError ocip_API ocipFFTForward(ocipProgram, ocipBuffer, ocipBuffer)
{
   return CL_INVALID_OPERATION;
}
ocipError ocip_API ocipFFTInverse(ocipProgram, ocipBuffer, ocipBuffer)
{
   return CL_INVALID_OPERATION;
}


ocipError ocip_API ocipPrepareImageProximityFFT(ocipProgram *, ocipBuffer, ocipBuffer )
{
   return CL_INVALID_OPERATION;
}

ocipError ocip_API ocipSqrDistanceFFT(ocipProgram , ocipBuffer , ocipBuffer , ocipBuffer)
{
   return CL_INVALID_OPERATION;
}

ocipError ocip_API ocipSqrDistanceFFT_Norm(ocipProgram , ocipBuffer , ocipBuffer , ocipBuffer)
{
   return CL_INVALID_OPERATION;
}
ocipError ocip_API ocipCrossCorrFFT(ocipProgram , ocipBuffer , ocipBuffer , ocipBuffer)
{
   return CL_INVALID_OPERATION;
}
ocipError ocip_API ocipCrossCorrFFT_Norm(ocipProgram , ocipBuffer , ocipBuffer , ocipBuffer)
{
   return CL_INVALID_OPERATION;
}

#endif // USE_CLFFT


// Helpers
SProgramList& GetList()
{
   COpenCL * Context = g_CurrentContext;
   
   if (Context == nullptr)
      throw cl::Error(CL_INVALID_CONTEXT, "No current context - use ocipInitialize() first");

   if (g_ProgramList[Context] == nullptr)
      throw cl::Error(CL_INVALID_CONTEXT, "Invalid current context - ocipInitialize()");

   return *g_ProgramList[Context];
}
