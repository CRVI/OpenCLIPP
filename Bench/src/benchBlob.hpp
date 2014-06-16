////////////////////////////////////////////////////////////////////////////////
//! @file	: benchBlob.hpp
//! @date   : Apr 2014
//!
//! @brief  : Benchmark class for blob labeling
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

class BlobBench : public IBench1in0out
{
public:
   BlobBench()
   :  m_CLBufferDst(nullptr),
      m_Program(nullptr)
   { }

   bool CompareCL(BlobBench * This);

   void RunIPP();
   void RunCL();
   void RunNPP();
   
   bool HasCVTest() const { return false; }
   bool HasNPPTest() const { return false; }

   void Create(uint Width, uint Height);

   void Free();

protected:
   CSimpleImage m_ImgDstIPP;
   CSimpleImage m_ImgDstCL;
   CSimpleImage m_ImgDstNPP;
   CSimpleImage m_ImgDstCV;

   ocipBuffer m_CLBufferDst;

   ocipProgram m_Program;

   vector<unsigned char> m_IPPBuffer;

   void * m_NPPDst;
   int m_NPPDstStep;

   CV_CODE(oclMat m_CVDst);
};

void BlobBench::Create(uint Width, uint Height)
{
   IBench1in0out::Create<unsigned char>(Width, Height);

   // Make m_ImgSrc all black
   for (uint y = 0; y < Height; y++)
      for (uint x = 0; x < Width; x++)
         *m_ImgSrc.Data(x, y) = 0;

   // Make white rectangles in the image
   for (int i = 0; i < 5; i++)
   {
      uint StartX = rand() % Width;
      uint EndX = rand() % (Width - StartX) + StartX;
      uint StartY = rand() % Height;
      uint EndY = rand() % (Height - StartY) + StartY;
      for (uint y = StartY; y < EndY; y++)
         for (uint x = StartX; x < EndX; x++)
            *m_ImgSrc.Data(x, y) = 255;
   }

   // Also make a circle
   int Diam = min(Width, Height) / 4;
   int MiddleX = rand() % (Width - Diam * 2) + Diam;
   int MiddleY = rand() % (Height - Diam * 2) + Diam;

   for (int y = 0; y < int(Height); y++)
      for (int x = 0; x < int(Width); x++)
         if (sqrt((x - MiddleX) * (x - MiddleX) + (y - MiddleY) * (y - MiddleY)) <= Diam)
            *m_ImgSrc.Data(x, y) = 255;

   // Send the images to the GPU again
   ocipSendImageBuffer(m_CLBufferSrc);

   NPP_CODE(
      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   CV_CODE(
      m_CVSrc.upload(toMat(m_ImgSrc));
      )


   m_ImgDstIPP.Create<unsigned char>(Width, Height);
   m_ImgDstCL.Create<int>(Width, Height);
   m_ImgDstNPP.Create<int>(Width, Height);
   m_ImgDstCV.Create<int>(Width, Height);

   // Copy source to m_ImgDstIPP
   for (uint y = 0; y < Height; y++)
      for (uint x = 0; x < Width; x++)
         *m_ImgDstIPP.Data(x, y) = *m_ImgSrc.Data(x, y);

   ocipCreateImageBuffer(&m_CLBufferDst, m_ImgDstCL, m_ImgDstCL.Data(), CL_MEM_READ_WRITE);

   // NPP
   NPP_CODE(
      m_ImgDstNPP.Create<int>(Width, Height);
      m_NPPDst = NPP_Malloc<int>(Width, Height, m_NPPDstStep);
      )

   // OpenCV
   CV_CODE(
      m_ImgDstCV.Create<int>(Width, Height);
      m_CVDst.create(Height, Width, GetCVType<int>(1));
      )

   IPP_CODE(
      int BufSize = 0;
      ippiLabelMarkersGetBufferSize_8u_C1R(m_IPPRoi, &BufSize);
      m_IPPBuffer.resize(BufSize);
      )

   ocipPrepareBlob(&m_Program, m_CLBufferSrc);
}

void BlobBench::Free()
{
   IBench1in0out::Free();

   ocipReleaseImageBuffer(m_CLBufferDst);

   ocipReleaseProgram(m_Program);

   NPP_CODE(nppiFree(m_NPPDst);)

   CV_CODE( m_CVDst.release(); )
}

bool BlobBench::CompareCL(BlobBench * This)
{
   ocipAddC_V(m_CLBufferDst, m_CLBufferDst, 1);
   ocipReadImageBuffer(m_CLBufferDst);

   CSimpleImage DstIPP;
   DstIPP.Create<int>(m_ImgDstIPP.Width, m_ImgDstIPP.Height);

   IPP_CODE(
      ippiConvert_8u32s_C1R(m_ImgDstIPP.Data(), m_ImgDstIPP.Step, (Ipp32s*) DstIPP.Data(), DstIPP.Step, m_IPPRoi);
      )

   return CompareImages(m_ImgDstCL, DstIPP, m_ImgSrc, *this);
}

//-----------------------------------------------------------------------------------------------------------------------------
void BlobBench::RunIPP()
{
   IPP_CODE(
      int Num = 0;
      ippiLabelMarkers_8u_C1IR(m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi, 1, 254, ippiNormInf, &Num, m_IPPBuffer.data());   // In place
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
void BlobBench::RunCL()
{
   ocipComputeLabels(m_Program, m_CLBufferSrc, m_CLBufferDst, 4);
   ocipRenameLabels(m_Program, m_CLBufferDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
void BlobBench::RunNPP()
{
   //NPP_CODE(
      // NPP does not have LabelMarkers
      // It has GraphCut instead which works differently
      //)
}
