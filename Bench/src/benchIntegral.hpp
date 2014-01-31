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

class IntegralBench : public IBench1in1out
{
public:
   IntegralBench()
   :  m_Program(nullptr)
   { }

   void Create(uint Width, uint Height);
   void Free();
   void RunIPP();
   void RunCUDA();
   void RunCL();
   void RunNPP();

   bool CompareCUDA(IntegralBench * This);
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
void IntegralBench::Create(uint Width, uint Height)
{
   // IPP & NPP require Dst to be 1 pixel larger and taller than Src
   IBench1in1out::Create<unsigned char, float>(Width, Height, Width + 1, Height + 1);

   // Re-allocate with proper size for CL - which require same size images
   m_ImgDstCL.Create<float>(Width, Height);
   ocipReleaseImage(m_CLDst);
   ocipCreateImage(&m_CLDst, m_ImgDstCL.ToSImage(), m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   ocipPrepareIntegral(&m_Program, m_CLSrc);

   CUDA_CODE(
      CUDAPP(Scan_Init)(Width, Height, SCAN_AXIS_BOTH);

      // Re-allocate with proper size and type for CUDA - requires float for Src and same size images
      CUDAPP(Free)(m_CUDASrc);
      CUDAPP(Free)(m_CUDADst);

      // This CUDA library requires a float input image for integral scan
      CSimpleImage FloatImage;
      FloatImage.Create<float>(Width, Height);

      // Use OpenCLIPP to convert to a float image
      ocipImage CLFloatImage = nullptr;
      ocipCreateImage(&CLFloatImage, FloatImage.ToSImage(), FloatImage.Data(), CL_MEM_READ_WRITE);

      ocipConvert(m_CLSrc, CLFloatImage);

      ocipReadImage(CLFloatImage);
      ocipReleaseImage(CLFloatImage);

      // Send to CUDA device memory
      CUDAPP(Malloc<float>)((float*&) m_CUDASrc, m_CUDASrcStep, Width, Height);
      CUDAPP(Upload<float>)((float*) FloatImage.Data(), FloatImage.Step,
         (float*) m_CUDASrc, m_CUDASrcStep, m_ImgSrc.Width, m_ImgSrc.Height);

      // Allocate Dst with same size as input
      CUDAPP(Malloc<float>)((float*&) m_CUDADst, m_CUDADstStep, Width, Height);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void IntegralBench::Free()
{
   IBench1in1out::Free();

   CUDA_CODE(CUDAPP(Scan_Shutdown)();)

   ocipReleaseProgram(m_Program);
}
//-----------------------------------------------------------------------------------------------------------------------------
void IntegralBench::RunIPP()
{
   IPP_CODE(ippiIntegral_8u32f_C1R(m_ImgSrc.Data(), m_ImgSrc.Step, (Ipp32f*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi, 0);)
}
//-----------------------------------------------------------------------------------------------------------------------------
void IntegralBench::RunCL()
{
   ocipIntegralScan(m_Program, m_CLSrc, m_CLDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void IntegralBench::RunNPP()
{
   NPP_CODE(nppiIntegral_8u32f_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp32f*) m_NPPDst, m_NPPDstStep, m_NPPRoi, 0);)
}
//-----------------------------------------------------------------------------------------------------------------------------
void IntegralBench::RunCUDA()
{
   CUDA_CODE(
      CUDAPP(Scan_32f_C1)((float *) m_CUDASrc, m_CUDASrcStep, (float *) m_CUDADst, m_CUDADstStep,
         m_ImgSrc.Width, m_ImgSrc.Height, CUDA_OP_ADD, SCAN_AXIS_BOTH);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
bool IntegralBench::CompareCUDA(IntegralBench * This)
{
   //Download the CUDA buffer into an host equivalent
   CSimpleImage CUDADst(m_ImgDstCL.ToSImage());

   CUDA_CODE(
      CUDAPP(Download)((float*) m_CUDADst, m_CUDADstStep, (float*) CUDADst.Data(), CUDADst.Step, 
         CUDADst.Width, CUDADst.Height);
      )

   // CUDA version is missing the bottom and right lines
   CImageROI ROI(m_ImgDstIPP, 0, 0, m_ImgDstCL.Width - 1, m_ImgDstCL.Height - 1);

   return CompareImages(CUDADst, ROI, m_ImgSrc, *This);
}
//-----------------------------------------------------------------------------------------------------------------------------
bool IntegralBench::CompareCL(IntegralBench * This)
{
   ocipReadImage(m_CLDst);

   // IPP results has a 1px added black line on the left and top of the image
   CImageROI ROI(m_ImgDstIPP, 1, 1, m_ImgDstCL.Width, m_ImgDstCL.Height);

   return CompareImages(m_ImgDstCL, ROI, m_ImgSrc, *This);
}
