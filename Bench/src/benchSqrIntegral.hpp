////////////////////////////////////////////////////////////////////////////////
//! @file	: benchSqrIntegral.hpp
//! @date   : Mar 2014
//!
//! @brief  : Benchmark class for square integral
//! 
//! Copyright (C) 2014 - CRVI
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

template<typename DataType> class SqrIntegralBench;

typedef SqrIntegralBench<float>   CONCATENATE(SqrIntegralBench, F32);
typedef SqrIntegralBench<double>  CONCATENATE(SqrIntegralBench, F64);

template<typename DataType>
class SqrIntegralBench : public IBench1in1out
{
public:

   SqrIntegralBench(): IBench1in1out(USE_BUFFER),
                 m_Program(nullptr) 
   { }

   void RunIPP();
   //void RunNPP();
   void RunCL();
   void RunCV();

   bool HasCUDATest() const { return false; }

   void Create(uint Width, uint Height);
   void Free();

   bool CompareCL(SqrIntegralBench * This);

   float CompareTolerance() const
   {
      return 0.005f;
   }

   bool CompareTolRelative() const
   {
      return true;
   }

protected:
   ocipProgram m_Program;
   CSimpleImage m_IppIntegral;
};

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void SqrIntegralBench<DataType>::Create(uint Width, uint Height)
{
   // IPP & NPP require Dst to be 1 pixel larger and taller than Src
   IBench1in1out::Create<unsigned char, double>(Width, Height, Width + 1, Height + 1);

   // Re-allocate with proper size for CL - which require same size images
   m_ImgDstCL.Create<DataType>(Width, Height);

   m_IppIntegral.Create<float>(m_ImgDstIPP.Width, m_ImgDstIPP.Height, m_ImgDstIPP.Channels);

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
void SqrIntegralBench<DataType>::Free()
{
   IBench1in1out::Free();

   m_IppIntegral.Free();
   ocipReleaseProgram(m_Program);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void SqrIntegralBench<DataType>::RunIPP()
{
   IPP_CODE(ippiSqrIntegral_8u32f64f_C1R(m_ImgSrc.Data(), m_ImgSrc.Step, (Ipp32f*) m_IppIntegral.Data(), m_IppIntegral.Step, (Ipp64f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi, 0, 0);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void SqrIntegralBench<float>::RunCL()
{
   if (m_UsesBuffer)
   {
      ocipSqrIntegral_B(m_Program, m_CLBufferSrc, m_CLBufferDst);
   }
   else
   {
      ocipSqrIntegral(m_Program, m_CLSrc, m_CLDst);
   }
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void SqrIntegralBench<double>::RunCL()
{
   if (m_UsesBuffer)
   {
      ocipSqrIntegral_B(m_Program, m_CLBufferSrc, m_CLBufferDst);
   }
   else
   {
      //There's no function for output image type of F64(double)
   }
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void SqrIntegralBench<DataType>::RunCV()
{
   CV_CODE( 
      oclMat Dummy;
      integral(m_CVSrc, Dummy, m_CVDst); )    // m_CVDst will be converted to F32
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
bool SqrIntegralBench<DataType>::CompareCL(SqrIntegralBench * This)
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
