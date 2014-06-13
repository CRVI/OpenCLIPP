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

template<typename DataType, int FactorX = 7, int FactorY = 3, int Interpolation = ocipNearestNeighbour>
class ResizeBench;

template<typename DataType, int Interpolation = ocipNearestNeighbour>
class ResizeBiggerBench : public ResizeBench<DataType, 13, 17, Interpolation>
{ };

template<typename DataType>
class ResizeLinearBench : public ResizeBench<DataType, 7, 3, ocipLinear>
{ };

template<typename DataType>
class ResizeBiggerLinearBench : public ResizeBench<DataType, 13, 17, ocipLinear>
{ };

template<typename DataType>
class ResizeCubicBench : public ResizeBench<DataType, 7, 3, ocipCubic>
{ };

template<typename DataType>
class ResizeBiggerCubicBench : public ResizeBench<DataType, 13, 17, ocipCubic>
{ };

template<typename DataType>
class ResizeLanczosBench : public ResizeBench<DataType, 7, 3, ocipLanczos3>
{ };

template<typename DataType>
class ResizeBiggerLanczosBench : public ResizeBench<DataType, 13, 17, ocipLanczos3>
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

protected:

   void SetDstSize(uint Width, uint Height, ocipInterpolationType Interpolation)
   {
      m_DstSize.Width = Width;
      m_DstSize.Height = Height;
      m_Interpolation = Interpolation;
   }

   ocipInterpolationType m_Interpolation;

   IPP_CODE(
      unsigned char * m_IPPBuffer;
      IppiSize m_IPPDstSize;

      int GetIPPMode() const
      {
         switch (m_Interpolation)
         {
         case ocipNearestNeighbour:
            return ippNearest;
         case ocipLinear:
            return ippLinear;
         case ocipCubic:
            return IPPI_INTER_CUBIC;
         case ocipLanczos3:
            return IPPI_INTER_LANCZOS;
         case ocipSuperSampling:
            return IPPI_INTER_SUPER;
         default:
            return ippNearest;
         }
      } )

   NPP_CODE(
      NppiRect m_NPPSrcROI;
      NppiSize m_NPPDstROI;

      NppiInterpolationMode GetNPPMode() const
      {
         switch (m_Interpolation)
         {
         case ocipNearestNeighbour:
            return NPPI_INTER_NN;
         case ocipLinear:
            return NPPI_INTER_LINEAR;
         case ocipCubic:
            return NPPI_INTER_CUBIC;
         case ocipLanczos3:
            return NPPI_INTER_LANCZOS;
         case ocipSuperSampling:
            return NPPI_INTER_SUPER;
         default:
            return NPPI_INTER_NN;
         }
      } )

   CV_CODE(
      int GetCVMode() const
      {
         switch (m_Interpolation)
         {
         case ocipNearestNeighbour:
            return INTER_NEAREST;
         case ocipLinear:
            return INTER_LINEAR;
         case ocipCubic:
            return INTER_CUBIC;
         case ocipLanczos3:
            return INTER_LANCZOS4;
         case ocipSuperSampling:
            return INTER_AREA;
         default:
            return INTER_NEAREST;
         }
      } )

   SSize m_DstSize;
};

template<typename DataType>
void ResizeBenchBase<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, DataType>(Width, Height, m_DstSize.Width, m_DstSize.Height);

   IPP_CODE(__ID(
      m_IPPDstSize.width = m_DstSize.Width;
      m_IPPDstSize.height = m_DstSize.Height;
      IppiRect ippSrcROI = {0, 0, Width, Height};
      IppiRect DstROI = {0, 0, m_DstSize.Width, m_DstSize.Height};
      int BufSize = 0;
      ippiResizeGetBufSize(ippSrcROI, DstROI, 1, GetIPPMode(), &BufSize);
      m_IPPBuffer = new unsigned char[BufSize];
   ))

   NPP_CODE(__ID(
      NppiRect SrcROI = {0, 0, Width, Height};
      m_NPPSrcROI = SrcROI;
      m_NPPDstROI.width = m_DstSize.Width;
      m_NPPDstROI.height = m_DstSize.Height;
   ))
}

template<typename DataType>
void ResizeBenchBase<DataType>::Free()
{
   IBench1in1out::Free();

   IPP_CODE(
      delete [] m_IPPBuffer;
   )
}

template<typename DataType>
void ResizeBenchBase<DataType>::RunCL()
{
   if (m_UsesBuffer)
      ocipResize_V(m_CLBufferSrc, m_CLBufferDst, m_Interpolation, false);
   else
      ocipResize(m_CLSrc, m_CLDst, m_Interpolation, false);
}

template<typename DataType>
void ResizeBenchBase<DataType>::RunCV()
{
   CV_CODE( resize(m_CVSrc, m_CVDst, m_CVDst.size(), 0, 0, GetCVMode()); )
}

template<>
void ResizeBenchBase<unsigned char>::RunIPP()
{
   IPP_CODE(__ID(
      IppiSize SrcSize = {m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect SrcROI = {0, 0, m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect DstROI = {0, 0, m_ImgDstIPP.Width, m_ImgDstIPP.Height};
      double XFactor = DstROI.width * 1. / SrcROI.width;
      double YFactor = DstROI.height * 1. / SrcROI.height;
      ippiResizeSqrPixel_8u_C1R(
         m_ImgSrc.Data(), SrcSize, m_ImgSrc.Step, SrcROI, 
         m_ImgDstIPP.Data(), m_ImgDstIPP.Step, DstROI, 
         XFactor, YFactor, 0, 0, GetIPPMode(), m_IPPBuffer);
   ))
}

template<>
void ResizeBenchBase<unsigned short>::RunIPP()
{
   IPP_CODE(__ID(
      IppiSize SrcSize = {m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect SrcROI = {0, 0, m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect DstROI = {0, 0, m_ImgDstIPP.Width, m_ImgDstIPP.Height};
      double XFactor = DstROI.width * 1. / SrcROI.width;
      double YFactor = DstROI.height * 1. / SrcROI.height;
      ippiResizeSqrPixel_16u_C1R(
         (Ipp16u*) m_ImgSrc.Data(), SrcSize, m_ImgSrc.Step, SrcROI, 
         (Ipp16u*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, DstROI, 
         XFactor, YFactor, 0, 0, GetIPPMode(), m_IPPBuffer);
   ))
}


template<>
void ResizeBenchBase<float>::RunIPP()
{
   IPP_CODE(__ID(
      IppiSize SrcSize = {m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect SrcROI = {0, 0, m_ImgSrc.Width, m_ImgSrc.Height};
      IppiRect DstROI = {0, 0, m_ImgDstIPP.Width, m_ImgDstIPP.Height};
      double XFactor = DstROI.width * 1. / SrcROI.width;
      double YFactor = DstROI.height * 1. / SrcROI.height;
      ippiResizeSqrPixel_32f_C1R(
         (Ipp32f*) m_ImgSrc.Data(), SrcSize, m_ImgSrc.Step, SrcROI, 
         (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, DstROI, 
         XFactor, YFactor, 0, 0, GetIPPMode(), m_IPPBuffer);       
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

template<typename DataType, int FactorX, int FactorY, int Interpolation>
class ResizeBench : public ResizeBenchBase<DataType>
{
public:
   void Create(uint Width, uint Height)
   {
      this->SetDstSize(Width * FactorX / 10, Height * FactorY / 10, ocipInterpolationType(Interpolation));
      ResizeBenchBase<DataType>::Create(Width, Height);
   }

   float CompareTolerance() const
   {
      if (Interpolation == ocipCubic && is_same<DataType, unsigned char>::value)
         return 2;   // There are minor differences in the bicubic results

      if (Interpolation == ocipLanczos2 || Interpolation == ocipLanczos3)
         return 0.15f;   // Allow higher tolerance for Lanczos

      return .05f;   // High tolerance to allow for minor interpolation differences
   }

   bool CompareTolRelative() const
   {
      if (Interpolation == ocipCubic && is_same<DataType, unsigned char>::value)
         return false;

      return true;
   }
};
