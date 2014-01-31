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
   : m_CUDASrc(nullptr)
   , m_CUDASrcStep(0)
   , m_CLBufferSrc(nullptr)
   , m_CLBufferDst(nullptr)
   { }

   void Create(uint Width, uint Height);
   void Free();

   void RunIPP();
   void RunCUDA();
   void RunCL();

   bool HasNPPTest() { return false; } // NPP will have the same speed as CUDA

   bool CompareCUDA(TransferBench*) { return true; }
   bool CompareCL(TransferBench*) { return true; }

protected:
   CSimpleImage m_ImgSrc;
   CSimpleImage m_ImgDst;

   unsigned char * m_CUDASrc;
   unsigned char * m_CUDADst;
   uint  m_CUDASrcStep;
   uint  m_CUDADstStep;

   ocipBuffer m_CLBufferSrc;
   ocipBuffer m_CLBufferDst;
};
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::Create(uint Width, uint Height)
{
   m_ImgSrc.Create<unsigned char>(Width, Height);
   m_ImgDst.Create<unsigned char>(Width, Height);
   FillRandomImg(m_ImgSrc);

   // CUDA
   CUDA_CODE(
      CUDAPP(Malloc)((unsigned char *&) m_CUDASrc, m_CUDASrcStep, Width, Height);
      CUDAPP(Malloc)((unsigned char *&) m_CUDADst, m_CUDADstStep, Width, Height);
      )

   // CL
   ocipCreateImageBuffer(&m_CLBufferSrc, m_ImgSrc.ToSImage(), m_ImgSrc.Data(), CL_MEM_READ_ONLY);
   ocipCreateImageBuffer(&m_CLBufferDst, m_ImgSrc.ToSImage(), m_ImgSrc.Data(), CL_MEM_WRITE_ONLY);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::Free()
{
   // CUDA
   CUDA_CODE(
      CUDAPP(Free)(m_CUDASrc);
      CUDAPP(Free)(m_CUDADst);
      )

   // CL
   ocipReleaseImageBuffer(m_CLBufferSrc);
   ocipReleaseImageBuffer(m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunIPP()
{
   // Nothing to do
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunCL()
{
   ocipSendImageBuffer(m_CLBufferSrc);
   ocipReadImageBuffer(m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void TransferBench::RunCUDA()
{
   CUDA_CODE(
      CUDAPP(Upload)(m_ImgSrc.Data(), m_ImgSrc.Step,
          m_CUDASrc, m_CUDASrcStep, m_ImgSrc.Width, m_ImgSrc.Height);

      CUDAPP(Download)(m_CUDADst, m_CUDADstStep, m_ImgDst.Data(),
         m_ImgDst.Step, m_ImgDst.Width, m_ImgDst.Height);
      )
}
