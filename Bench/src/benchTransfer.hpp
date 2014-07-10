////////////////////////////////////////////////////////////////////////////////
//! @file	: benchTransfer.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for device memory <-> host transfer
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

class TransferBench : public IBench
{
public:
   TransferBench()
   : m_NPPSrc(nullptr)
   , m_NPPDst(nullptr)
   , m_NPPSrcStep(0)
   , m_NPPDstStep(0)
   , m_CLBufferSrc(nullptr)
   , m_CLBufferDst(nullptr)
   { }

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunNPP();
   void RunCL();

   bool HasCVTest() { return false; }

   bool CompareNPP(TransferBench*) { return true; }
   bool CompareCL(TransferBench*) { return true; }

protected:
   CSimpleImage m_ImgSrc;
   CSimpleImage m_ImgDst;

   unsigned char * m_NPPSrc;
   unsigned char * m_NPPDst;
   int  m_NPPSrcStep;
   int  m_NPPDstStep;

   ocipImage m_CLBufferSrc;
   ocipImage m_CLBufferDst;
};
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::Create(uint Width, uint Height)
{
   m_ImgSrc.Create<unsigned char>(Width, Height);
   m_ImgDst.Create<unsigned char>(Width, Height);
   FillRandomImg(m_ImgSrc);

   // NPP
   NPP_CODE(
      m_NPPSrc = (unsigned char*) NPP_Malloc<unsigned char>(Width, Height, m_NPPSrcStep);
      m_NPPDst = (unsigned char*) NPP_Malloc<unsigned char>(Width, Height, m_NPPDstStep);
      )

   // CL
   ocipCreateImage(&m_CLBufferSrc, m_ImgSrc.ToSImage(), m_ImgSrc.Data(), CL_MEM_READ_ONLY);
   ocipCreateImage(&m_CLBufferDst, m_ImgSrc.ToSImage(), m_ImgSrc.Data(), CL_MEM_WRITE_ONLY);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::Free()
{
   // NPP
   NPP_CODE(
      nppiFree(m_NPPSrc);
      nppiFree(m_NPPDst);
      )

   // CL
   ocipReleaseImage(m_CLBufferSrc);
   ocipReleaseImage(m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunIPP()
{
   // Nothing to do
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunCL()
{
   ocipSendImage(m_CLBufferSrc);
   ocipReadImage(m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunNPP()
{
   NPP_CODE(
      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), m_ImgSrc.Height, cudaMemcpyHostToDevice);

      cudaMemcpy2D(m_ImgDst.Data(), m_ImgDst.Step, m_NPPDst, m_NPPDstStep,
         m_ImgDst.BytesWidth(), m_ImgDst.Height, cudaMemcpyDeviceToHost);
      )
}
