////////////////////////////////////////////////////////////////////////////////
//! @file	: benchFFT.hpp
//! @date   : Jan 2014
//!
//! @brief  : Benchmark classes for Fast Fourrier Transform
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


// log2 should be in the standard library by the Microsoft version of the standard library does not provide it
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( n ) / log( 2 );  
}

class FFTForwardBench : public IBench1in1out
{
public:
   FFTForwardBench()
   : IBench1in1out(true)
   { }

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunCL();
   void RunCV();

   bool HasNPPTest() const { return false; }
   bool HasCUDATest() const { return false; }

   float CompareTolerance() const { return 0.005f; }

private:
   ocipProgram m_Program;

   IPP_CODE(
      IppiFFTSpec_R_32f * m_IPPSpec;
      Ipp8u * m_IPPBuffer;
      CImage<float> m_IPPPacked;
      CImage<float> m_IPPUnpacked;
   )
};

class FFTBackwardBench : public IBench1in1out
{
public:
   FFTBackwardBench()
   : IBench1in1out(true)
   { }

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunCL();
   void RunCV();

   bool HasNPPTest() const { return false; }
   bool HasCUDATest() const { return false; }

   float CompareTolerance() const { return 0.005f; }

private:
   ocipProgram m_Program;

   IPP_CODE(
      IppiFFTSpec_R_32f * m_IPPSpec;
      Ipp8u * m_IPPBuffer;
      CImage<float> m_IPPPacked;
   )
};

void FFTForwardBench::Create(uint Width, uint Height)
{
   // Source image
   IBench1in0out::Create<float>(Width, Height);

   // Destination image
   uint DstWidth = Width / 2 + 1;
   uint DstHeight = Height;

   // CPU
   m_ImgDstIPP.Create<float>(DstWidth, DstHeight, 2);

   // CL
   m_ImgDstCL.Create<float>(DstWidth, DstHeight, 2);

   // OpenCV OCL
   m_ImgDstCV.Create<float>(DstWidth, DstHeight, 2);

   if (m_UsesBuffer)
      ocipCreateImageBuffer(&m_CLBufferDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);
   else
      ocipCreateImage(&m_CLDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(
      m_ImgDstNPP.Create<float>(DstWidth, DstHeight, 2);
      m_NPPDst = NPP_Malloc<sizeof(float)>(DstWidth * 2, DstHeight, m_NPPDstStep);
      )

   // CUDA
   CUDA_CODE(
      CUDAPP(Malloc<float>)((float*&) m_CUDADst, m_CUDADstStep, DstWidth, DstHeight);
      )


   // Prepare other resources
   ocipPrepareFFT(&m_Program, m_CLBufferSrc, m_CLBufferDst);

   IPP_CODE(
      int OrderX = (int) Log2(Width);
      int OrderY = (int) Log2(Height);

      if (pow(2, OrderX) != Width || pow(2, OrderY) != Height)
      {
         printf("\nippiFFT accepts only images that have a size that is a power of 2\n");
         return;
      }

      ippiFFTInitAlloc_R_32f(&m_IPPSpec, OrderX, OrderY, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone);

      int BufferSize = 0;
      ippiFFTGetBufSize_R_32f(m_IPPSpec, &BufferSize);

      m_IPPBuffer = (Ipp8u*) ippMalloc(BufferSize);

      m_IPPPacked.Create<float>(Width, Height);

      m_IPPUnpacked.Create<float>(Width, Height, 2);
      )
}

void FFTForwardBench::Free()
{
   IBench1in1out::Free();

   // Free other resources
   ocipReleaseProgram(m_Program);

   IPP_CODE(
      ippiFFTFree_R_32f(m_IPPSpec);
      m_IPPSpec = nullptr;

      ippFree(m_IPPBuffer);
      m_IPPBuffer = nullptr;

      m_IPPPacked.Free();
      m_IPPUnpacked.Free();
      )
}


void FFTBackwardBench::Create(uint Width, uint Height)
{
   // Source image
   uint SrcWidth = Width / 2 + 1;
   m_ImgSrc.Create<float>(SrcWidth, Height, 2);
   FillRandomImg(m_ImgSrc);


   // CL
   if (m_UsesBuffer)
   {
      ocipCreateImageBuffer(&m_CLBufferSrc, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_ONLY);
      ocipSendImageBuffer(m_CLBufferSrc);
   }
   else
   {
      ocipCreateImage(&m_CLSrc, m_ImgSrc, m_ImgSrc.Data(), CL_MEM_READ_ONLY);
      ocipSendImage(m_CLSrc);
   }

   // IPP
   IPP_CODE(
      m_IPPRoi.width = SrcWidth;
      m_IPPRoi.height = Height;
      )

   // NPP
   NPP_CODE(
      m_NPPSrc = NPP_Malloc<sizeof(float)>(SrcWidth * 2, Height, m_NPPSrcStep);
      m_NPPRoi.width = SrcWidth;
      m_NPPRoi.height = Height;

      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   // CUDA
   CUDA_CODE(
      CUDAPP(Malloc<float>)((float*&) m_CUDASrc, m_CUDASrcStep, SrcWidth * 2, Height);
      CUDAPP(Upload<float>)((float*) m_ImgSrc.Data(), m_ImgSrc.Step,
         (float*) m_CUDASrc, m_CUDASrcStep, m_ImgSrc.Width * 2, m_ImgSrc.Height);
      )

   // CV
   CV_CODE(
      m_CVSrc.create(SrcWidth, Width, GetCVType<float>(2));
      m_CVSrc.upload(toMat(m_ImgSrc));
      )


   // Destination image
   uint DstWidth = Width;
   uint DstHeight = Height;

   // CPU
   m_ImgDstIPP.Create<float>(DstWidth, DstHeight);

   // CL
   m_ImgDstCL.Create<float>(DstWidth, DstHeight);

   if (m_UsesBuffer)
      ocipCreateImageBuffer(&m_CLBufferDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);
   else
      ocipCreateImage(&m_CLDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(
      m_ImgDstNPP.Create<float>(DstWidth, DstHeight);
      m_NPPDst = NPP_Malloc<sizeof(float)>(DstWidth , DstHeight, m_NPPDstStep);
      )

   // CUDA
   CUDA_CODE(
      CUDAPP(Malloc<float>)((float*&) m_CUDADst, m_CUDADstStep, DstWidth, DstHeight);
      )

   // OpenCV
   CV_CODE(
      m_ImgDstCV.Create<float>(DstWidth, DstHeight);
      m_CVDst.create(DstHeight, DstWidth, GetCVType<float>(1));
      )

   // Prepare other resources
   ocipPrepareFFT(&m_Program, m_CLBufferDst, m_CLBufferSrc);

   IPP_CODE(
      int OrderX = (int) Log2(Width);
      int OrderY = (int) Log2(Height);

      if (pow(2, OrderX) != Width || pow(2, OrderY) != Height)
      {
         printf("\nippiFFT accepts only images that have a size that is a power of 2\n");
         return;
      }

      ippiFFTInitAlloc_R_32f(&m_IPPSpec, OrderX, OrderY, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone);

      int BufferSize = 0;
      ippiFFTGetBufSize_R_32f(m_IPPSpec, &BufferSize);

      m_IPPBuffer = (Ipp8u*) ippMalloc(BufferSize);

      m_IPPPacked.Create<float>(Width, Height);
      )


}

void FFTBackwardBench::Free()
{
   IBench1in1out::Free();

   // Free other resources
   ocipReleaseProgram(m_Program);

   IPP_CODE(
      ippiFFTFree_R_32f(m_IPPSpec);
      m_IPPSpec = nullptr;

      ippFree(m_IPPBuffer);
      m_IPPBuffer = nullptr;

      m_IPPPacked.Free();
      )
}

void FFTForwardBench::RunIPP()
{
   IPP_CODE(
      ippiFFTFwd_RToPack_32f_C1R((float*) m_ImgSrc.Data(), m_ImgSrc.Step, (float*) m_IPPPacked.Data(), m_IPPPacked.Step, m_IPPSpec, m_IPPBuffer);

      // NOTE : This conversion could be done during comparison instead
      ippiPackToCplxExtend_32f32fc_C1R((float*) m_IPPPacked.Data(), m_IPPRoi, m_IPPPacked.Step, (Ipp32fc*) m_IPPUnpacked.Data(), m_IPPUnpacked.Step);

      IppiSize CopyRoi = m_IPPRoi;
      CopyRoi.width += 2;

      ippiCopy_32f_C1R((float*) m_IPPUnpacked.Data(), m_IPPUnpacked.Step, (float*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, CopyRoi);
      )
}

void FFTForwardBench::RunCL()
{
   ocipFFTForward(m_Program, m_CLBufferSrc, m_CLBufferDst);
}

void FFTForwardBench::RunCV()
{
   CV_CODE( dft(m_CVSrc, m_CVDst); )
}

void FFTBackwardBench::RunIPP()
{
   IPP_CODE(
      IppiSize Roi = m_IPPRoi;
      Roi.width = m_IPPPacked.Width;

      ippiCplxExtendToPack_32fc32f_C1R((Ipp32fc*) m_ImgSrc.Data(), m_ImgSrc.Step, Roi, (float*) m_IPPPacked.Data(), m_IPPPacked.Step);

      ippiFFTInv_PackToR_32f_C1R((float*) m_IPPPacked.Data(), m_IPPPacked.Step, (float*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPSpec, m_IPPBuffer);
      )
}

void FFTBackwardBench::RunCL()
{
   ocipFFTInverse(m_Program, m_CLBufferSrc, m_CLBufferDst);
}

void FFTBackwardBench::RunCV()
{
   CV_CODE( dft(m_CVSrc, m_CVDst, m_CVDst.size(), DFT_INVERSE | DFT_REAL_OUTPUT); )
}
