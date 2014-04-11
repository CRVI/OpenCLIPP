////////////////////////////////////////////////////////////////////////////////
//! @file	: benchIntegral.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for integral scan
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

template<typename DataType> class IntegralBench;

typedef IntegralBench<float>   CONCATENATE(IntegralBench, F32);
typedef IntegralBench<double>  CONCATENATE(IntegralBench, F64);

template<typename DataType>
class IntegralBench : public IBench1in1out
{
public:
   IntegralBench()
   :  IBench1in1out(USE_BUFFER),
      m_Program(nullptr)
   { }

   void Create(uint Width, uint Height);
   void Free();
   void RunIPP();
   void RunCL();
   void RunCV();
   void RunNPP();

   bool CompareCL(IntegralBench * This);

   float CompareTolerance() const
   {
      // Compute an acceptable tolerance
      double ApproxSum = m_ImgDstCL.Width * 128. * m_ImgDstCL.Height;
      return float(ApproxSum / 100000);
   }

protected:
   ocipProgram m_Program;
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::Create(uint Width, uint Height)
{
   // IPP & NPP require Dst to be 1 pixel larger and taller than Src
   IBench1in1out::Create<unsigned char, float>(Width, Height, Width + 1, Height + 1);

   // Re-allocate with proper size for CL - which require same size images
   m_ImgDstCL.Create<DataType>(Width, Height);

   if (m_UsesBuffer)
   {
      ocipReleaseImageBuffer(m_CLBufferDst);
      ocipCreateImageBuffer(&m_CLBufferDst, m_ImgDstCL.ToSImage(), m_ImgDstCL.Data(), CL_MEM_READ_WRITE);
      ocipPrepareImageBufferIntegral(&m_Program, m_CLBufferSrc);
   }
   else
   {
      if (is_same<DataType, float>::value)
      {
         ocipReleaseImage(m_CLDst);
         ocipCreateImage(&m_CLDst, m_ImgDstCL.ToSImage(), m_ImgDstCL.Data(), CL_MEM_READ_WRITE);
         ocipPrepareIntegral(&m_Program, m_CLSrc);
      }
      else
      {
         // F64 images are not supported in images
      }

   }

}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::Free()
{
   IBench1in1out::Free();

   ocipReleaseProgram(m_Program);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::RunIPP()
{
   IPP_CODE(ippiIntegral_8u32f_C1R(this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, 0);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::RunCL()
{
   if (m_UsesBuffer)
   {
      ocipIntegral_B(m_Program, m_CLBufferSrc, m_CLBufferDst);
   }
   else
   {
      if (is_same<DataType, double>::value)  //There's no function for output image type of F64(double)
         return;

      ocipIntegral(m_Program, m_CLSrc, m_CLDst);
   }
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::RunCV()
{
   CV_CODE( integral(m_CVSrc, m_CVDst); )    // This does not work well, it recreates m_CVDst as a 32S image...
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void IntegralBench<DataType>::RunNPP()
{
   NPP_CODE(nppiIntegral_8u32f_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp32f*) m_NPPDst, m_NPPDstStep, m_NPPRoi, 0);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
bool IntegralBench<DataType>::CompareCL(IntegralBench * This)
{
   if (m_UsesBuffer)
   {
      ocipReadImageBuffer(m_CLBufferDst);
   }
   else
   {
      if (is_same<DataType, double>::value)
         return false;  // F64 is not supported in images

      ocipReadImage(m_CLDst);
   }

   // IPP results has a 1px added black line on the left and top of the image
   CImageROI ROI(m_ImgDstIPP, 1, 1, m_ImgDstCL.Width, m_ImgDstCL.Height);

   return CompareImages(m_ImgDstCL, ROI, m_ImgSrc, *This);
}
