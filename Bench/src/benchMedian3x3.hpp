////////////////////////////////////////////////////////////////////////////////
//! @file	: benchMedian3x3.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for 3x3 median filter
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

class Median3x3Bench : public BenchUnaryBase<unsigned char, true>
{
public:
   Median3x3Bench()
   : m_MaskSize(3, 3)
   , m_MaskAnchor(1, 1)
   { }

   void RunIPP();
   void RunCUDA();
   void RunCL();
   void RunCV();

   bool HasNPPTest() { return false; }

   SSize CompareSize() const { return m_MaskSize; }
   SPoint CompareAnchor() const { return m_MaskAnchor; }

private:
   SSize m_MaskSize;
   SPoint m_MaskAnchor;
};
//-----------------------------------------------------------------------------------------------------------------------------
void Median3x3Bench::RunIPP()
{
   IPP_CODE(
      IppiSize ROI;
      ROI.width = m_ImgSrc.Width - 2;
      ROI.height = m_ImgSrc.Height - 2;

      ippiFilterMedian_8u_C1R(m_ImgSrc.Data(1, 1), m_ImgSrc.Step, m_ImgDstIPP.Data(1, 1), m_ImgDstIPP.Step, ROI,
         *(IppiSize*)&m_MaskSize, *(IppiPoint*)&m_MaskAnchor);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void Median3x3Bench::RunCL()
{
   if (CLUsesBuffer())
      ocipMedian_V(m_CLBufferSrc, m_CLBufferDst, 3);
   else
      ocipMedian(m_CLSrc, m_CLDst, 3);
}
//-----------------------------------------------------------------------------------------------------------------------------
void Median3x3Bench::RunCUDA()
{
   CUDA_CODE(
      CUDAPP(FilterMedian_3x3)(
         (unsigned char*) m_CUDASrc, m_CUDASrcStep,
         (unsigned char*) m_CUDADst, m_CUDADstStep,
         m_ImgSrc.Width, m_ImgSrc.Height);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void Median3x3Bench::RunCV()
{
   CV_CODE( medianFilter(m_CVSrc, m_CVDst, 3); )
}
