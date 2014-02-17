////////////////////////////////////////////////////////////////////////////////
//! @file	: benchMorphoBase.hpp
//! @date   : Jul 2013
//!
//! @brief  : Base class for morphology benchmarks
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

#pragma once

#define MORPHO_USES_BUFFER true

class MorphoBenchBase : public IBench1in1out
{
public:
   MorphoBenchBase()
   : IBench1in1out(MORPHO_USES_BUFFER)
   , m_CUDATmp(nullptr)
   , m_CUDATmpStep(0)
   , m_NPPTmp(nullptr)
   , m_NPPTmpStep(0)
   , m_MaskSize(3, 3)
   , m_MaskSize2(5, 5)
   , m_MaskAnchor(1, 1)
   , m_MaskAnchor2(2, 2)
   { }

   void Create(uint Width, uint Height);
   void Free();

   SSize CompareSize() const { return m_MaskSize2; }
   SPoint CompareAnchor() const { return m_MaskAnchor2; }

protected:

   CSimpleImage m_ImgTemp;

   ocipImage m_CLTmp;
   ocipBuffer m_CLBufferTmp;

   unsigned char* m_CUDATmp;
   uint m_CUDATmpStep;

   unsigned char * m_NPPTmp;
   int m_NPPTmpStep;

   SSize m_MaskSize, m_MaskSize2;
   SPoint m_MaskAnchor, m_MaskAnchor2;

   IPP_CODE(
      IppiSize m_ROI1;
      IppiSize m_ROI2;
      )
};

inline void MorphoBenchBase::Create(uint Width, uint Height)
{
   IBench1in1out::Create<unsigned char, unsigned char>(Width, Height);

   //V( (BinarizeImg<unsigned char, 1>(m_ImgSrc)) );

   m_ImgTemp.Create<unsigned char>(Width, Height, 1);

   // IPP
   IPP_CODE(
      m_ROI1.width = m_IPPRoi.width - 2;
      m_ROI1.height = m_IPPRoi.height - 2;
      m_ROI2.width = m_IPPRoi.width - 4;
      m_ROI2.height = m_IPPRoi.height - 4;
      )

   // CUDA
   CUDA_CODE(CUDAPP(Malloc_8u_C1)(m_CUDATmp, m_CUDATmpStep, Width, Height);)

   // CL
   if (m_UsesBuffer)
      ocipCreateImageBuffer(&m_CLBufferTmp, m_ImgSrc.ToSImage(), nullptr, CL_MEM_READ_WRITE);
   else
      ocipCreateImage(&m_CLTmp, m_ImgSrc.ToSImage(), nullptr, CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(m_NPPTmp = (Npp8u*) NPP_Malloc<1>(Width, Height, m_NPPTmpStep);)
}

inline void MorphoBenchBase::Free()
{
   IBench1in1out::Free();

   CUDA_CODE(CUDAPP(Free)(m_CUDATmp);)

   NPP_CODE(nppiFree(m_NPPTmp);)
}

// Erode and Dilate benches
#define BENCH_NAME Erode
#define CV_NAME erode
#include "benchMorpho.hpp"

#define BENCH_NAME Dilate
#define CV_NAME dilate
#include "benchMorpho.hpp"
