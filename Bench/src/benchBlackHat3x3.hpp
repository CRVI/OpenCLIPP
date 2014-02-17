////////////////////////////////////////////////////////////////////////////////
//! @file	: benchBlackHat3x3.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for BlackHat3x3
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

class BlackHatBench : public MorphoBenchBase
{
public:
   void RunIPP();
   void RunCUDA();
   void RunCL();
   void RunNPP();
   void RunCV();
};
//-----------------------------------------------------------------------------------------------------------------------------
void BlackHatBench::RunIPP()
{
   IPP_CODE(
      // Close (Dilate then Erode)
      ippiDilate3x3_8u_C1R(m_ImgSrc.Data(1, 1), m_ImgSrc.Step, m_ImgDstIPP.Data(1, 1), m_ImgDstIPP.Step, m_ROI1);
      ippiErode3x3_8u_C1R(m_ImgDstIPP.Data(2, 2), m_ImgDstIPP.Step, m_ImgTemp.Data(2, 2), m_ImgTemp.Step, m_ROI2);

      // Close(Tmp) - Src
      ippiSub_8u_C1RSfs(m_ImgSrc.Data(), m_ImgSrc.Step, m_ImgTemp.Data(), m_ImgTemp.Step, 
         m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi, 0);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void BlackHatBench::RunCL()
{
   if (m_UsesBuffer)
      ocipBlackHat_B(m_CLBufferSrc, m_CLBufferDst, m_CLBufferTmp, 1, 3);
   else
      ocipBlackHat(m_CLSrc, m_CLDst, m_CLTmp, 1, 3);
}
//-----------------------------------------------------------------------------------------------------------------------------
void BlackHatBench::RunCUDA()
{
   CUDA_CODE(
      CUDAPP(Blackhat_8u_C1)((unsigned char*) m_CUDASrc,
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
void BlackHatBench::RunNPP()
{
   NPP_CODE(
      //Blackhat
      nppiDilate3x3_8u_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi);
      nppiErode3x3_8u_C1R((Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPTmp, m_NPPTmpStep, m_NPPRoi);
   
      // Close(Tmp) - Src
      nppiSub_8u_C1RSfs((Npp8u*) m_NPPSrc, m_NPPSrcStep, m_NPPTmp, m_NPPTmpStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi, 0);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void BlackHatBench::RunCV()
{
   CV_CODE(morphologyEx(m_CVSrc, m_CVDst, CV_MOP_BLACKHAT, Mat());)
}
