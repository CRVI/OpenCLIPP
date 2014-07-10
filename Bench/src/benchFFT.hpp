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

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunNPP();
   void RunCL();
   void RunCV();

   float CompareTolerance() const { return 0.005f; }

private:
   ocipProgram m_Program;

   IPP_CODE(
      IppiFFTSpec_R_32f * m_IPPSpec;
      Ipp8u * m_IPPBuffer;
      CImage<float> m_IPPPacked;
      CImage<float> m_IPPUnpacked;
   )

   NPP_CODE( cufftHandle m_NPPPlan; )
};

class FFTBackwardBench : public IBench1in1out
{
public:

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunNPP();
   void RunCL();
   void RunCV();

   float CompareTolerance() const { return 0.005f; }

   bool CompareTolRelative() const { return true; }

private:
   ocipProgram m_Program;

   IPP_CODE(
      IppiFFTSpec_R_32f * m_IPPSpec;
      Ipp8u * m_IPPBuffer;
      CImage<float> m_IPPPacked;
   )

   NPP_CODE( cufftHandle m_NPPPlan; )
};

void FFTForwardBench::Create(uint Width, uint Height)
{
   // Destination image size
   uint DstWidth = Width / 2 + 1;
   uint DstHeight = Height;

   // Create source and destination images
   IBench1in1out::Create<float, float>(Width, Height, DstWidth, DstHeight, true, 1, 2);


   // Prepare other resources
   ocipPrepareFFT(&m_Program, m_CLSrc, m_CLDst);

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

   NPP_CODE( cufftPlan2d(&m_NPPPlan, Width, Height, CUFFT_R2C); )
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

   NPP_CODE( cufftDestroy(m_NPPPlan); )
}


void FFTBackwardBench::Create(uint Width, uint Height)
{
   // Source image size
   uint SrcWidth = Width / 2 + 1;

   // Create source and destination images
   IBench1in1out::Create<float, float>(SrcWidth, Height, Width, Height, true, 2, 1);


   // Prepare other resources
   ocipPrepareFFT(&m_Program, m_CLDst, m_CLSrc);

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

   NPP_CODE( cufftPlan2d(&m_NPPPlan, Width, Height, CUFFT_C2R); )

   // Make a forward FFT to m_ImgSrc
   FillRandomImg(m_ImgDstCL);
   ocipFFTForward(m_Program, m_CLDst, m_CLSrc);
   ocipReadImage(m_CLSrc);

   bool TestInversion = false;
   if (TestInversion)
   {
      // Test if we can get back the source image with the inverse transformation
      ocipFFTInverse(m_Program, m_CLSrc, m_CLDst);
      ocipDivC(m_CLDst, m_CLDst, float(Width * Height));
      ocipReadImage(m_CLDst);
      // m_ImgDstCL should be exaclty the same as it was before calling ocipFFTInverse()
   }

   // Send m_ImgSrc back to the GPU
   ocipSendImage(m_CLSrc);

   CV_CODE(
      m_CVSrc.upload(toMat(m_ImgSrc));
      )

   NPP_CODE(
      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
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

   NPP_CODE( cufftDestroy(m_NPPPlan); )
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

void FFTForwardBench::RunNPP()
{
   NPP_CODE(
      cufftExecR2C(m_NPPPlan, (cufftReal*) m_NPPSrc, (cufftComplex*) m_NPPDst);
      )
}

void FFTForwardBench::RunCL()
{
   ocipFFTForward(m_Program, m_CLSrc, m_CLDst);
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

void FFTBackwardBench::RunNPP()
{
   NPP_CODE(
      cufftExecC2R(m_NPPPlan, (cufftComplex*) m_NPPSrc, (cufftReal*) m_NPPDst);
      )
}

void FFTBackwardBench::RunCL()
{
   ocipFFTInverse(m_Program, m_CLSrc, m_CLDst);
}

void FFTBackwardBench::RunCV()
{
   CV_CODE( dft(m_CVSrc, m_CVDst, m_CVDst.size(), DFT_INVERSE | DFT_REAL_OUTPUT); )
}
