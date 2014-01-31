////////////////////////////////////////////////////////////////////////////////
//! @file	: benchTopHat3x3.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for TopHat3x3
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

class TopHatBench : public MorphoBenchBase
{
public:
   void RunIPP();
   void RunCUDA();
   void RunCL();
   void RunNPP();
};
//-----------------------------------------------------------------------------------------------------------------------------
void TopHatBench::RunIPP()
{
   IPP_CODE(
      // Open (Erode then Dilate)
      ippiErode3x3_8u_C1R(m_ImgSrc.Data(1, 1), m_ImgSrc.Step, m_ImgDstIPP.Data(1, 1), m_ImgDstIPP.Step, m_ROI1);
      ippiDilate3x3_8u_C1R(m_ImgDstIPP.Data(2, 2), m_ImgDstIPP.Step, m_ImgTemp.Data(2, 2), m_ImgTemp.Step, m_ROI2);

      // Src - Open(Tmp)
      ippiSub_8u_C1RSfs(m_ImgTemp.Data(), m_ImgTemp.Step, m_ImgSrc.Data(),
         m_ImgSrc.Step, m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi, 0);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void TopHatBench::RunCL()
{
   if (m_UsesBuffer)
      ocipTopHat_B(m_CLBufferSrc, m_CLBufferDst, m_CLBufferTmp, 1, 3);
   else
      ocipTopHat(m_CLSrc, m_CLDst, m_CLTmp, 1, 3);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TopHatBench::RunCUDA()
{
   CUDA_CODE(
      CUDAPP(Tophat_8u_C1)((unsigned char*) m_CUDASrc,
         m_CUDASrcStep,
         (unsigned char*) m_CUDATmp,
         m_CUDATmpStep,
         (unsigned char*) m_CUDADst,
         m_CUDADstStep,
         m_ImgSrc.Width,
         m_ImgSrc.Height,
         m_MaskSize.Width,
         m_MaskSize.Height,
         m_MaskAnchor.X,
         m_MaskAnchor.Y);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void TopHatBench::RunNPP()
{
   NPP_CODE(
      // Tophat
      nppiErode3x3_8u_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi);
      nppiDilate3x3_8u_C1R((Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPTmp, m_NPPTmpStep, m_NPPRoi);
   
      // Src - Open(Tmp)
      nppiSub_8u_C1RSfs(m_NPPTmp, m_NPPTmpStep, (Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi, 0);
      )
}
