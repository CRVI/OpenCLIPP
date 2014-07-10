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
   float CompareTolerance() const { return 0.0001f; }
   bool CompareTolRelative() const { return false; }
};

class IBench1in0out : public IBench
{
public:
   IBench1in0out()
   : m_CLBufferSrc(nullptr)
   , m_NPPSrc(nullptr)
   , m_NPPSrcStep(0)
   { }

   template<typename DataType> void Create(uint Width, uint Height, bool AllowNegative = true, int NbChannels = 1);
   void Free();

protected:

   CSimpleImage m_ImgSrc;

   ocipImage m_CLBufferSrc;

   void * m_NPPSrc;
   int m_NPPSrcStep;
   NPP_CODE(NppiSize m_NPPRoi;)

   IPP_CODE(IppiSize m_IPPRoi);

   CV_CODE(oclMat m_CVSrc);
};

class IBench1in1out : public IBench1in0out
{
public:
   IBench1in1out()
   : m_CLBufferDst(nullptr)
   , m_NPPDst(nullptr)
   , m_NPPDstStep(0)
   { }

   template<typename SrcType, typename DstType> void Create(
      uint Width, uint Height, uint DstWidth = 0, uint DstHeight = 0, bool AllowNegative = true, int NbChannelsSrc = 1, int NbChannelsDst = 1);

   void Free();
   template<class T> bool CompareCL(T * This);
   template<class T> bool CompareNPP(T * This);
   template<class T> bool CompareCV(T * This);

protected:
   CSimpleImage m_ImgDstIPP;
   CSimpleImage m_ImgDstCL;
   CSimpleImage m_ImgDstNPP;
   CSimpleImage m_ImgDstCV;

   ocipImage m_CLBufferDst;

   void * m_NPPDst;
   int m_NPPDstStep;

   CV_CODE(oclMat m_CVDst);
};

class IBench2in1out : public IBench1in1out
{
public:
   IBench2in1out()
   : m_CLBufferSrcB(nullptr)
   , m_NPPSrcB(nullptr)
   , m_NPPSrcBStep(0)
   { }

   template<typename SrcType, typename DstType> void Create(uint Width, uint Height);
   void Free();

protected:
   CSimpleImage m_ImgSrcB;

   ocipImage m_CLBufferSrcB;

   void * m_NPPSrcB;
   int m_NPPSrcBStep;

   CV_CODE(oclMat m_CVSrcB);
};

template<typename DataType>
class BenchUnaryBase : public IBench1in1out
{
public:
   void Create(uint Width, uint Height, int NbChannels = 1)
   {
      IBench1in1out::Create<DataType, DataType>(Width, Height,
         0, 0, true, NbChannels, NbChannels);
   }

};

template<typename DataType>
class BenchBinaryBase : public IBench2in1out
{
public:
   void Create(uint Width, uint Height)
   {
      IBench2in1out::Create<DataType, DataType>(Width, Height);
   }

};


NPP_CODE(
   template<typename T>
   void * NPP_Malloc(uint Width, uint Height, int& Step, int NbChannels = 1)
   {
      size_t element_width = sizeof(T) * NbChannels;

      switch (element_width)
      {
      case 1:
         return (void *) nppiMalloc_8u_C1(Width, Height, &Step);
      case 2:
         return (void *) nppiMalloc_16u_C1(Width, Height, &Step);
      case 4:
         return (void *) nppiMalloc_32s_C1(Width, Height, &Step);
      case 8:
         return (void *) nppiMalloc_16s_C4(Width, Height, &Step);
      case 16:
         return (void *) nppiMalloc_32s_C4(Width, Height, &Step);
      default:
         return nullptr;
      }

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

   template<>
   int GetCVType<double>(int NbChannels)
   {
      return CV_MAKETYPE(CV_64F, NbChannels);
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
      case SImage::F64:
         return GetCVType<double>(NbChannels);
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
inline void IBench1in0out::Create(uint Width, uint Height, bool AllowNegative, int NbChannels)
{
   // Source image
   m_ImgSrc.Create<DataType>(Width, Height, NbChannels);
   FillRandomImg(m_ImgSrc);

   if (!AllowNegative)
   {
      // remove negative values 
      ocipImage Buffer = nullptr;
      ocipCreateImage(&Buffer, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_WRITE);
      ocipAbs(Buffer, Buffer);
      ocipReadImage(Buffer);
      ocipReleaseImage(Buffer);
   }

   // CL
   ocipCreateImage(&m_CLBufferSrc, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_WRITE);
   ocipSendImage(m_CLBufferSrc);

   // IPP
   IPP_CODE(
      m_IPPRoi.width = Width;
      m_IPPRoi.height = Height;
      )

   // NPP
   NPP_CODE(
      m_NPPSrc = NPP_Malloc<DataType>(Width, Height, m_NPPSrcStep, NbChannels);
      m_NPPRoi.width = Width;
      m_NPPRoi.height = Height;

      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   // OpenCV
   CV_CODE(
      m_CVSrc.create(Height, Width, GetCVType<DataType>(NbChannels));
      m_CVSrc.upload(toMat(m_ImgSrc));
      )
}

inline void IBench1in0out::Free()
{
   NPP_CODE(nppiFree(m_NPPSrc);)

   ocipReleaseImage(m_CLBufferSrc);

   CV_CODE( m_CVSrc.release(); )
}

template<typename SrcType, typename DstType>
inline void IBench1in1out::Create(uint Width, uint Height, uint DstWidth, uint DstHeight, bool AllowNegative, int NbChannelsSrc, int NbChannelsDst)
{
   IBench1in0out::Create<SrcType>(Width, Height, AllowNegative, NbChannelsSrc);

   if (DstWidth == 0)
      DstWidth = Width;

   if (DstHeight == 0)
      DstHeight = Height;

   // CPU
   m_ImgDstIPP.Create<DstType>(DstWidth, DstHeight, NbChannelsDst);

   // CL
   m_ImgDstCL.Create<DstType>(DstWidth, DstHeight, NbChannelsDst);

   ocipCreateImage(&m_CLBufferDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(
      m_ImgDstNPP.Create<DstType>(DstWidth, DstHeight, NbChannelsDst);
      m_NPPDst = NPP_Malloc<DstType>(DstWidth, DstHeight, m_NPPDstStep, NbChannelsDst);
      )

   // OpenCV
   CV_CODE(
      m_ImgDstCV.Create<DstType>(DstWidth, DstHeight, NbChannelsDst);
      m_CVDst.create(DstHeight, DstWidth, GetCVType<DstType>(NbChannelsDst));
      )
}

inline void IBench1in1out::Free()
{
   IBench1in0out::Free();

   NPP_CODE(nppiFree(m_NPPDst);)

   ocipReleaseImage(m_CLBufferDst);

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
   ocipCreateImage(&m_CLBufferSrcB, m_ImgSrcB, m_ImgSrcB.Data(), CL_MEM_READ_ONLY);
   ocipSendImage(m_CLBufferSrcB);

   // NPP
   NPP_CODE(
      m_NPPSrcB = NPP_Malloc<SrcType>(Width, Height, m_NPPSrcBStep);
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

   ocipReleaseImage(m_CLBufferSrcB);

   CV_CODE( m_CVSrcB.release(); )
}

template<class T> 
inline bool IBench1in1out::CompareCL(T * This)
{
   ocipReadImage(m_CLBufferDst);

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
