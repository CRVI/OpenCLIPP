////////////////////////////////////////////////////////////////////////////////
//! @file	: benchMorpho.hpp
//! @date   : Jul 2013
//!
//! @brief  : Creates a benchmark class for morphology operations
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

class CONCATENATE(BENCH_NAME, Bench) : public MorphoBenchBase
{
public:
   void RunIPP();
   void RunCL();
   void RunNPP();
   void RunCV();
};
//-----------------------------------------------------------------------------------------------------------------------------
void CONCATENATE(BENCH_NAME, Bench)::RunIPP()
{
   IPP_CODE(
      CONCATENATE(CONCATENATE(ippi, BENCH_NAME), 3x3_8u_C1R)(
         m_ImgSrc.Data(1, 1), m_ImgSrc.Step, m_ImgDstIPP.Data(1, 1), m_ImgDstIPP.Step, m_ROI1);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void CONCATENATE(BENCH_NAME, Bench)::RunCL()
{
   if (m_UsesBuffer)
      CONCATENATE(CONCATENATE(ocip, BENCH_NAME), _B)(m_CLBufferSrc, m_CLBufferDst, 3);
   else
      CONCATENATE(ocip, BENCH_NAME)(m_CLSrc, m_CLDst, 3);
}
//-----------------------------------------------------------------------------------------------------------------------------
void CONCATENATE(BENCH_NAME, Bench)::RunNPP()
{
   NPP_CODE(
      CONCATENATE(CONCATENATE(nppi, BENCH_NAME), 3x3_8u_C1R)(
         (Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp8u*) m_NPPDst, m_NPPDstStep, m_NPPRoi);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void CONCATENATE(BENCH_NAME, Bench)::RunCV()
{
   CV_CODE(CV_NAME(m_CVSrc, m_CVDst, Mat());)
}

#undef CV_NAME
#undef BENCH_NAME
