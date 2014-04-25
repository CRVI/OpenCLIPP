////////////////////////////////////////////////////////////////////////////////
//! @file	: benchResize.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for image resizing
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

template<typename DataType, int FactorX = 7, int FactorY = 3, bool LinearInterpolation = false>
class ResizeBench;

template<typename DataType, bool LinearInterpolation = false>
class ResizeBiggerBench : public ResizeBench<DataType, 13, 17, LinearInterpolation>
{ };

template<typename DataType>
class ResizeLinearBench : public ResizeBench<DataType, 7, 3, true>
{ };

template<typename DataType>
class ResizeBiggerLinearBench : public ResizeBench<DataType, 13, 17, true>
{ };

// FactorX and FactorX are in 1/10th, so 10 will mean same size
template<typename DataType>
class ResizeBenchBase : public IBench1in1out
{
public:
   ResizeBenchBase()
   :  IBench1in1out(USE_BUFFER)
   { }
   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunNPP();
   void RunCL();
   void RunCV();

   float CompareTolerance() const { return .05f; }    // High tolerance to allow for minor interpolation differences
   bool CompareTolRelative() const { return true; }

protected:

   void Init();

   void SetDstSize(uint Width, uint Height, bool LinearInterpolation)
   {
      m_DstSize.Width = Width;
      m_DstSize.Height = Height;
      m_LinearInterpolation = LinearInterpolation;
   }

   bool m_LinearInterpolation;

   IPP_CODE(
      unsigned char * m_IPPBuffer;
      IppiSize m_IPPDstSize;
      IppiResizeSpec_32f * m_ResizeSpec;

      IppiInterpolationType GetIPPMode()
      {
         if (m_LinearInterpolation)
            return ippLinear;

         return ippNearest;
      } )

   NPP_CODE(
      NppiRect m_NPPSrcROI;
      NppiSize m_NPPDstROI;

      NppiInterpolationMode GetNPPMode()
      {
         if (m_LinearInterpolation)
            return NPPI_INTER_LINEAR;

         return NPPI_INTER_NN;
      } )

   SSize m_DstSize;
};

template<typename DataType>
void ResizeBenchBase<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, DataType>(Width, Height, m_DstSize.Width, m_DstSize.Height);

   IPP_CODE(
      m_IPPDstSize.width = m_DstSize.Width;
      m_IPPDstSize.height = m_DstSize.Height;
      m_ResizeSpec = nullptr;
   )

   Init();

   NPP_CODE(__ID(
      NppiRect SrcROI = {0, 0, Width, Height};
      m_NPPSrcROI = SrcROI;
      m_NPPDstROI.width = m_DstSize.Width;
      m_NPPDstROI.height = m_DstSize.Height;
   ))
}

template<>
void ResizeBenchBase<unsigned char>::Init()
{
   IPP_CODE(
      int SpecSize = 0;
      int InitSize = 0;
      ippiResizeGetSize_8u(m_IPPRoi, m_IPPDstSize, GetIPPMode(), 0, &SpecSize, &InitSize);
      m_ResizeSpec = (IppiResizeSpec_32f*) new unsigned char[SpecSize];

      if (m_LinearInterpolation)
         ippiResizeLinearInit_8u(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);
      else
         ippiResizeNearestInit_8u(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);

      int BufSize = 0;
      ippiResizeGetBufferSize_8u(m_ResizeSpec, m_IPPDstSize, 1, &BufSize);

      m_IPPBuffer = new unsigned char[BufSize];
   )
}

template<>
void ResizeBenchBase<unsigned short>::Init()
{
   IPP_CODE(
      int SpecSize = 0;
      int InitSize = 0;
      ippiResizeGetSize_16u(m_IPPRoi, m_IPPDstSize, GetIPPMode(), 0, &SpecSize, &InitSize);
      m_ResizeSpec = (IppiResizeSpec_32f*) new unsigned char[SpecSize];

      if (m_LinearInterpolation)
         ippiResizeLinearInit_16u(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);
      else
         ippiResizeNearestInit_16u(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);

      int BufSize = 0;
      ippiResizeGetBufferSize_16u(m_ResizeSpec, m_IPPDstSize, 1, &BufSize);

      m_IPPBuffer = new unsigned char[BufSize];
   )
}

template<>
void ResizeBenchBase<float>::Init()
{
   IPP_CODE(
      int SpecSize = 0;
      int InitSize = 0;
      ippiResizeGetSize_32f(m_IPPRoi, m_IPPDstSize, GetIPPMode(), 0, &SpecSize, &InitSize);
      m_ResizeSpec = (IppiResizeSpec_32f*) new unsigned char[SpecSize];

      if (m_LinearInterpolation)
         ippiResizeLinearInit_32f(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);
      else
         ippiResizeNearestInit_32f(m_IPPRoi, m_IPPDstSize, m_ResizeSpec);

      int BufSize = 0;
      ippiResizeGetBufferSize_32f(m_ResizeSpec, m_IPPDstSize, 1, &BufSize);

      m_IPPBuffer = new unsigned char[BufSize];
   )
}

template<typename DataType>
void ResizeBenchBase<DataType>::Free()
{
   IBench1in1out::Free();

   IPP_CODE(
      delete [] m_IPPBuffer;
      delete [] (unsigned char*) m_ResizeSpec;
   )
}

template<typename DataType>
void ResizeBenchBase<DataType>::RunCL()
{
   if (m_UsesBuffer)
      ocipResize_V(m_CLBufferSrc, m_CLBufferDst, m_LinearInterpolation, false);
   else
      ocipResize(m_CLSrc, m_CLDst, m_LinearInterpolation, false);
}

template<typename DataType>
void ResizeBenchBase<DataType>::RunCV()
{
   CV_CODE( resize(m_CVSrc, m_CVDst, m_CVDst.size(), 0, 0, (m_LinearInterpolation ? INTER_LINEAR : INTER_NEAREST)); )
}

template<>
void ResizeBenchBase<unsigned char>::RunIPP()
{
   IPP_CODE(__ID(
      IppiPoint Offset = {0, 0};
      if (m_LinearInterpolation)
         ippiResizeLinear_8u_C1R(
            m_ImgSrc.Data(), m_ImgSrc.Step,
            m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize,
            ippBorderRepl, nullptr,
            m_ResizeSpec, m_IPPBuffer);
      else
         ippiResizeNearest_8u_C1R(
            m_ImgSrc.Data(), m_ImgSrc.Step,
            m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize, m_ResizeSpec, m_IPPBuffer);
   ))
}

template<>
void ResizeBenchBase<unsigned short>::RunIPP()
{
   IPP_CODE(__ID(
      IppiPoint Offset = {0, 0};
      if (m_LinearInterpolation)
         ippiResizeLinear_16u_C1R(
            (Ipp16u*) m_ImgSrc.Data(), m_ImgSrc.Step,
            (Ipp16u*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize,
            ippBorderRepl, nullptr,
            m_ResizeSpec, m_IPPBuffer);
      else
         ippiResizeNearest_16u_C1R(
            (Ipp16u*) m_ImgSrc.Data(), m_ImgSrc.Step,
            (Ipp16u*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize, m_ResizeSpec, m_IPPBuffer);
   ))
}


template<>
void ResizeBenchBase<float>::RunIPP()
{
   IPP_CODE(__ID(
      IppiPoint Offset = {0, 0};
      if (m_LinearInterpolation)
         ippiResizeLinear_32f_C1R(
            (Ipp32f*) m_ImgSrc.Data(), m_ImgSrc.Step,
            (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize,
            ippBorderRepl, nullptr,
            m_ResizeSpec, m_IPPBuffer);
      else
         ippiResizeNearest_32f_C1R(
            (Ipp32f*) m_ImgSrc.Data(), m_ImgSrc.Step,
            (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, Offset, m_IPPDstSize, m_ResizeSpec, m_IPPBuffer);          
   ))
}

template<>
void ResizeBenchBase<unsigned char>::RunNPP()
{
   NPP_CODE(
      nppiResize_8u_C1R(
         (Npp8u *) m_NPPSrc, m_NPPRoi, m_NPPSrcStep, m_NPPSrcROI,
         (Npp8u *) m_NPPDst, m_NPPDstStep, m_NPPDstROI,
         double(m_DstSize.Width) / m_NPPRoi.width,
         double(m_DstSize.Height) / m_NPPRoi.height, GetNPPMode());
   )
}

template<>
void ResizeBenchBase<unsigned short>::RunNPP()
{
   NPP_CODE(
      nppiResize_16u_C1R(
         (Npp16u *) m_NPPSrc, m_NPPRoi, m_NPPSrcStep, m_NPPSrcROI,
         (Npp16u *) m_NPPDst, m_NPPDstStep, m_NPPDstROI,
         double(m_DstSize.Width) / m_NPPRoi.width,
         double(m_DstSize.Height) / m_NPPRoi.height, GetNPPMode());
   )
}


template<>
void ResizeBenchBase<float>::RunNPP()
{
   NPP_CODE(
      nppiResize_32f_C1R(
         (Npp32f *) m_NPPSrc, m_NPPRoi, m_NPPSrcStep, m_NPPSrcROI,
         (Npp32f *) m_NPPDst, m_NPPDstStep, m_NPPDstROI,
         double(m_DstSize.Width) / m_NPPRoi.width,
         double(m_DstSize.Height) / m_NPPRoi.height, GetNPPMode());
   )
}

template<typename DataType, int FactorX, int FactorY, bool LinearInterpolation>
class ResizeBench : public ResizeBenchBase<DataType>
{
public:
   void Create(uint Width, uint Height)
   {
      this->SetDstSize(Width * FactorX / 10, Height * FactorY / 10, LinearInterpolation);
      ResizeBenchBase<DataType>::Create(Width, Height);
   }
};
