////////////////////////////////////////////////////////////////////////////////
//! @file	: benchBase.hpp
//! @date   : Jul 2013
//!
//! @brief  : Base benchmark classes - takes care of allocating and transferring images
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

#include "CImage.h"

class ICLBench
{
public:
   bool HasCLTest() const { return true; }
   void RunCL() { }
   template<class T> bool CompareCL(T * This) { return false; }
};

class INPPBench
{
public:
   bool HasNPPTest() const { return true; }
   void RunNPP() { }
   template<class T> bool CompareNPP(T * This) { return false; }
};

class ICVBench
{
public:
   bool HasCVTest() const { return true; }
   void RunCV() { }
   template<class T> bool CompareCV(T * This) { return false; }
};

class IBench : public ICLBench, public INPPBench, public ICVBench
{
public:
   SSize CompareSize() const { return SSize(1, 1); }
   SPoint CompareAnchor() const { return SPoint(0, 0); }
   float CompareTolerance() const { return SUCCESS_EPSILON; }
   bool CompareTolRelative() const { return false; }
};

class IBench1in0out : public IBench
{
public:
   IBench1in0out(bool CLUsesBuffer = false)
   : m_UsesBuffer(CLUsesBuffer)
   , m_CLSrc(nullptr)
   , m_CLBufferSrc(nullptr)
   , m_NPPSrc(nullptr)
   , m_NPPSrcStep(0)
   { }

   template<typename DataType> void Create(uint Width, uint Height, bool AllowNegative = true);
   void Free();
   bool CLUsesBuffer() const { return m_UsesBuffer; }

protected:

   bool m_UsesBuffer;

   CSimpleImage m_ImgSrc;

   ocipImage m_CLSrc;
   ocipBuffer m_CLBufferSrc;

   void * m_NPPSrc;
   int m_NPPSrcStep;
   NPP_CODE(NppiSize m_NPPRoi;)

   IPP_CODE(IppiSize m_IPPRoi);

   CV_CODE(oclMat m_CVSrc);
};

class IBench1in1out : public IBench1in0out
{
public:
   IBench1in1out(bool CLUsesBuffer = false)
   : IBench1in0out(CLUsesBuffer)
   , m_CLDst(nullptr)
   , m_CLBufferDst(nullptr)
   , m_NPPDst(nullptr)
   , m_NPPDstStep(0)
   { }

   template<typename SrcType, typename DstType> void Create(
      uint Width, uint Height, uint DstWidth = 0, uint DstHeight = 0, bool AllowNegative = true);

   void Free();
   template<class T> bool CompareCL(T * This);
   template<class T> bool CompareNPP(T * This);
   template<class T> bool CompareCV(T * This);

protected:
   CSimpleImage m_ImgDstIPP;
   CSimpleImage m_ImgDstCL;
   CSimpleImage m_ImgDstNPP;
   CSimpleImage m_ImgDstCV;

   ocipImage m_CLDst;
   ocipBuffer m_CLBufferDst;

   void * m_NPPDst;
   int m_NPPDstStep;

   CV_CODE(oclMat m_CVDst);
};

class IBench2in1out : public IBench1in1out
{
public:
   IBench2in1out(bool CLUsesBuffer = false)
   : IBench1in1out(CLUsesBuffer)
   , m_CLSrcB(nullptr)
   , m_CLBufferSrcB(nullptr)
   , m_NPPSrcB(nullptr)
   , m_NPPSrcBStep(0)
   { }

   template<typename SrcType, typename DstType> void Create(uint Width, uint Height);
   void Free();

protected:
   CSimpleImage m_ImgSrcB;

   ocipImage m_CLSrcB;
   ocipBuffer m_CLBufferSrcB;

   void * m_NPPSrcB;
   int m_NPPSrcBStep;

   CV_CODE(oclMat m_CVSrcB);
};

template<typename DataType, bool UseBuffer>
class BenchUnaryBase : public IBench1in1out
{
public:
   BenchUnaryBase()
   : IBench1in1out(UseBuffer)
   { }

   void Create(uint Width, uint Height)
   {
      IBench1in1out::Create<DataType, DataType>(Width, Height);
   }

   typedef DataType dataType;
};

template<typename DataType, bool UseBuffer>
class BenchBinaryBase : public IBench2in1out
{
public:
   BenchBinaryBase()
   : IBench2in1out(UseBuffer)
   { }

   void Create(uint Width, uint Height)
   {
      IBench2in1out::Create<DataType, DataType>(Width, Height);
   }

   typedef DataType dataType;
};

template <typename T>
struct SHasDataType {
   template <typename C> static char t(typename C::dataType*);
   template <typename> static int t(...);
   static const bool value = sizeof(t<T>(0)) == sizeof(char);
};

template<class Bench, bool HasType>
struct SGetBenchDataTypeImpl
{
   typedef unsigned char Type;
};

template<class Bench>
struct SGetBenchDataTypeImpl<Bench, true>
{
   typedef typename Bench::dataType Type;
};

template<class Bench>
struct SGetBenchDataType
{
   typedef typename SGetBenchDataTypeImpl<Bench, SHasDataType<Bench>::value>::Type Type;
};

NPP_CODE(template<int T>
   void * NPP_Malloc(uint Width, uint Height, int& Step)
   {
      return (void *) nppiMalloc_8u_C1(Width, Height, &Step);
   }

   template<>
   void * NPP_Malloc<2>(uint Width, uint Height, int& Step)
   {
      return (void *) nppiMalloc_16u_C1(Width, Height, &Step);
   }

   template<>
   void * NPP_Malloc<4>(uint Width, uint Height, int& Step)
   {
      return (void *) nppiMalloc_32s_C1(Width, Height, &Step);
   }
   )

CV_CODE(

   template<typename DataType>
   int GetCVType(int NbChannels)
   {
      assert(false);
      return 0;
   }

   template<>
   int GetCVType<unsigned char>(int NbChannels)
   {
      return CV_MAKETYPE(CV_8U, NbChannels);
   }

   template<>
   int GetCVType<char>(int NbChannels)
   {
      return CV_MAKETYPE(CV_8S, NbChannels);
   }

   template<>
   int GetCVType<unsigned short>(int NbChannels)
   {
      return CV_MAKETYPE(CV_16U, NbChannels);
   }

   template<>
   int GetCVType<short>(int NbChannels)
   {
      return CV_MAKETYPE(CV_16S, NbChannels);
   }

   template<>
   int GetCVType<int>(int NbChannels)
   {
      return CV_MAKETYPE(CV_32S, NbChannels);
   }

   template<>
   int GetCVType<float>(int NbChannels)
   {
      return CV_MAKETYPE(CV_32F, NbChannels);
   }

   int ToCVType(SImage::EDataType Type, int NbChannels = 1)
   {
      switch (Type)
      {
      case SImage::U8:
         return GetCVType<unsigned char>(NbChannels);
      case SImage::S8:
         return GetCVType<char>(NbChannels);
      case SImage::U16:
         return GetCVType<unsigned short>(NbChannels);
      case SImage::S16:
         return GetCVType<short>(NbChannels);
      case SImage::U32:
         return GetCVType<unsigned int>(NbChannels);
      case SImage::S32:
         return GetCVType<int>(NbChannels);
      case SImage::F32:
         return GetCVType<float>(NbChannels);
      default:
         assert(false);
         return 0;
      }
   }

   Mat toMat(const CSimpleImage& Img)
   {
      return Mat(Img.Height, Img.Width, ToCVType(Img.Type, Img.Channels), (void *) Img.Data(), Img.Step);
   }

   )

template<typename DataType> 
inline void IBench1in0out::Create(uint Width, uint Height, bool AllowNegative)
{
   // Source image
   m_ImgSrc.Create<DataType>(Width, Height);
   FillRandomImg(m_ImgSrc);

   if (!AllowNegative)
   {
      // remove negative values 
      ocipBuffer Buffer = nullptr;
      ocipCreateImageBuffer(&Buffer, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_WRITE);
      ocipAbs_V(Buffer, Buffer);
      ocipReadImageBuffer(Buffer);
      ocipReleaseImageBuffer(Buffer);
   }

   // CL
   if (m_UsesBuffer)
   {
      ocipCreateImageBuffer(&m_CLBufferSrc, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_WRITE);
      ocipSendImageBuffer(m_CLBufferSrc);
   }
   else
   {
      ocipCreateImage(&m_CLSrc, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_WRITE);
      ocipSendImage(m_CLSrc);
   }

   // IPP
   IPP_CODE(
      m_IPPRoi.width = Width;
      m_IPPRoi.height = Height;
      )

   // NPP
   NPP_CODE(
      m_NPPSrc = NPP_Malloc<sizeof(DataType)>(Width, Height, m_NPPSrcStep);
      m_NPPRoi.width = Width;
      m_NPPRoi.height = Height;

      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   // OpenCV
   CV_CODE(
      m_CVSrc.create(Height, Width, GetCVType<DataType>(1));
      m_CVSrc.upload(toMat(m_ImgSrc));
      )
}

inline void IBench1in0out::Free()
{
   NPP_CODE(nppiFree(m_NPPSrc);)

   ocipReleaseImageBuffer(m_CLBufferSrc);
   ocipReleaseImage(m_CLSrc);

   CV_CODE( m_CVSrc.release(); )
}

template<typename SrcType, typename DstType>
inline void IBench1in1out::Create(uint Width, uint Height, uint DstWidth, uint DstHeight, bool AllowNegative)
{
   IBench1in0out::Create<SrcType>(Width, Height, AllowNegative);

   if (DstWidth == 0)
      DstWidth = Width;

   if (DstHeight == 0)
      DstHeight = Height;

   // CPU
   m_ImgDstIPP.Create<DstType>(DstWidth, DstHeight);

   // CL
   m_ImgDstCL.Create<DstType>(DstWidth, DstHeight);

   if (m_UsesBuffer)
      ocipCreateImageBuffer(&m_CLBufferDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);
   else
      ocipCreateImage(&m_CLDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(
      m_ImgDstNPP.Create<DstType>(DstWidth, DstHeight);
      m_NPPDst = NPP_Malloc<sizeof(DstType)>(DstWidth, DstHeight, m_NPPDstStep);
      )

   // OpenCV
   CV_CODE(
      m_ImgDstCV.Create<DstType>(DstWidth, DstHeight);
      m_CVDst.create(DstHeight, DstWidth, GetCVType<DstType>(1));
      )
}

inline void IBench1in1out::Free()
{
   IBench1in0out::Free();

   NPP_CODE(nppiFree(m_NPPDst);)

   ocipReleaseImageBuffer(m_CLBufferDst);
   ocipReleaseImage(m_CLDst);

   CV_CODE( m_CVDst.release(); )
}

template<typename SrcType, typename DstType>
inline void IBench2in1out::Create(uint Width, uint Height)
{
   IBench1in1out::Create<SrcType, DstType>(Width, Height);

   // CPU
   m_ImgSrcB.Create<SrcType>(Width, Height);
   FillRandomImg(m_ImgSrcB, 1);

   // CL
   if (m_UsesBuffer)
   {
      ocipCreateImageBuffer(&m_CLBufferSrcB, m_ImgSrcB, m_ImgSrcB.Data(), CL_MEM_READ_ONLY);
      ocipSendImageBuffer(m_CLBufferSrcB);
   }
   else
   {
      ocipCreateImage(&m_CLSrcB, m_ImgSrcB, m_ImgSrcB.Data(), CL_MEM_READ_ONLY);
      ocipSendImage(m_CLSrcB);
   }

   // NPP
   NPP_CODE(
      m_NPPSrcB = NPP_Malloc<sizeof(SrcType)>(Width, Height, m_NPPSrcBStep);
      cudaMemcpy2D(m_NPPSrcB, m_NPPSrcBStep, m_ImgSrcB.Data(), m_ImgSrcB.Step,
         m_ImgSrcB.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   // OpenCV
   CV_CODE(
      m_CVSrcB.create(Height, Width, GetCVType<SrcType>(1));
      m_CVSrcB.upload(toMat(m_ImgSrcB));
      )
}

inline void IBench2in1out::Free()
{
   IBench1in1out::Free();

   NPP_CODE(nppiFree(m_NPPSrcB);)

   ocipReleaseImageBuffer(m_CLBufferSrcB);
   ocipReleaseImage(m_CLSrcB);

   CV_CODE( m_CVSrcB.release(); )
}

template<class T> 
inline bool IBench1in1out::CompareCL(T * This)
{
   if (m_UsesBuffer)
      ocipReadImageBuffer(m_CLBufferDst);
   else
      ocipReadImage(m_CLDst);

   return CompareImages(m_ImgDstCL, m_ImgDstIPP, m_ImgSrc, *This);
}

template<class T>
inline bool IBench1in1out::CompareNPP(T * This)
{
   NPP_CODE(
      cudaMemcpy2D(m_ImgDstNPP.Data(), m_ImgDstNPP.Step, m_NPPDst, m_NPPDstStep,
         m_ImgDstNPP.BytesWidth(), m_ImgDstNPP.Height, cudaMemcpyDeviceToHost);
   )

   return CompareImages(m_ImgDstNPP, m_ImgDstIPP, m_ImgSrc, *This);
}

template<class T>
inline bool IBench1in1out::CompareCV(T * This)
{
   CV_CODE(
      m_CVDst.download(toMat(m_ImgDstCV));
   )

   return CompareImages(m_ImgDstCV, m_ImgDstIPP, m_ImgSrc, *This);
}
