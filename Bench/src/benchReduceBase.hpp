////////////////////////////////////////////////////////////////////////////////
//! @file	: benchReduceBase.hpp
//! @date   : Jul 2013
//!
//! @brief  : Base class for reduction benchmarks
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

template<typename DataType, typename DstT>
class BenchReduceBase : public IBench1in0out
{
public:
   BenchReduceBase()
   : IBench1in0out(USE_BUFFER)
   , m_Program(nullptr)
   , m_IndxNPP(nullptr)
   , m_NPPWorkBuffer(nullptr)
   { }

   void Create(uint Width, uint Height, int NbChannels = 1);
   void Free();

   bool CompareCL(BenchReduceBase * This);
   bool CompareNPP(BenchReduceBase * This);
   bool CompareCV(BenchReduceBase * This);

   bool Compare(double * V1, DstT * V2);

   virtual float CompareTolerance() const { return SUCCESS_EPSILON; }

protected:

   ocipProgram m_Program;

   DstT m_DstIPP[4];
   SPoint m_IndxIPP;

   double m_DstCL[4];
   SPoint m_IndxCL;

   CV_CODE(Scalar m_DstCV;)
   CV_CODE(Scalar m_CVDummy;)
   CV_CODE(Point m_IndxCV;)
   CV_CODE(Point m_CVDummyIndx;)

   DstT * m_NPPDst;
   SPoint * m_IndxNPP;

   unsigned char* m_NPPWorkBuffer;

   friend class IBench1in0out;
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
void BenchReduceBase<DataType, DstT>::Create(uint Width, uint Height, int NbChannels)
{
   IBench1in0out::Create<DataType>(Width, Height, true, NbChannels);

   m_IndxCL = m_IndxIPP = SPoint(0, 0);

   CV_CODE( m_IndxCV = Point(0, 0); )

   if (CLUsesBuffer())
      ocipPrepareImageBufferStatistics(&m_Program, m_CLBufferSrc);
   else
      ocipPrepareStatistics(&m_Program, m_CLSrc);

   NPP_CODE(
      int BufferSize = 0;
      nppiMaxGetBufferHostSize_32f_C4R(m_NPPRoi, &BufferSize);
      cudaMalloc((void**) &m_NPPWorkBuffer, BufferSize);
      cudaMalloc((void**) &m_NPPDst, sizeof(DstT) * 4);
      cudaMalloc((void**) &m_IndxNPP, sizeof(m_IndxNPP));
      )

   if (std::is_same<float, DataType>::value)
   {
      // To prevent float values from overflowing, we divide the values to get them smaller
      if (this->m_UsesBuffer)
      {
         ocipDivC_V(m_CLBufferSrc, m_CLBufferSrc, 1000000);
         ocipReadImageBuffer(m_CLBufferSrc);
      }
      else
      {
         ocipDivC(m_CLSrc, m_CLSrc, 1000000);
         ocipReadImage(m_CLSrc);
      }
      
   }
   else if (std::is_same<unsigned char, DataType>::value)
   {
      // Remove smallest and biggest values
      if (this->m_UsesBuffer)
      {
         ocipThresholdGTLT_V(m_CLBufferSrc, m_CLBufferSrc, 2, 3, 253, 250);
         ocipReadImageBuffer(m_CLBufferSrc);
      }
      else
      {
         ocipThresholdGTLT(m_CLSrc, m_CLSrc, 2, 3, 253, 250);
         ocipReadImage(m_CLSrc);
      }

      // Place a single high and low pixel at a random location
      CImage<unsigned char>& Img = static_cast<CImage<unsigned char>&>(m_ImgSrc);
      Img(std::rand() % Width, std::rand() % Height) = 1;
      Img(std::rand() % Width, std::rand() % Height) = 254;
   }
   if (std::is_same<unsigned short, DataType>::value)
   {
      // Remove smallest and biggest values
      if (this->m_UsesBuffer)
      {
         ocipThresholdGTLT_V(m_CLBufferSrc, m_CLBufferSrc, 2, 3, 64000, 63200);
         ocipReadImageBuffer(m_CLBufferSrc);
      }
      else
      {
         ocipThresholdGTLT(m_CLSrc, m_CLSrc, 2, 3, 64000, 63200);
         ocipReadImage(m_CLSrc);
      }

      // Place a single high and low pixel at a random location
      CImage<unsigned short>& Img = static_cast<CImage<unsigned short>&>(m_ImgSrc);
      Img(std::rand() % Width, std::rand() % Height) = 1;
      Img(std::rand() % Width, std::rand() % Height) = 64539;
   }

   // Resend the image
   if (this->m_UsesBuffer)
      ocipSendImageBuffer(m_CLBufferSrc);
   else
      ocipSendImage(m_CLSrc);

   NPP_CODE(
      cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
         m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
      )

   CV_CODE(
      m_CVSrc.upload(toMat(m_ImgSrc));
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
void BenchReduceBase<DataType, DstT>::Free()
{
   IBench1in0out::Free();

   ocipReleaseProgram(m_Program);

   NPP_CODE(
      cudaFree(m_NPPWorkBuffer);
      cudaFree(m_NPPDst);
      cudaFree(m_IndxNPP);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareCL(BenchReduceBase *)
{
   if (m_IndxIPP.X != m_IndxCL.X)
      return false;
   
   if (m_IndxIPP.Y != m_IndxCL.Y)
      return false;

   return Compare(m_DstCL, m_DstIPP);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareNPP(BenchReduceBase *)
{
   DstT NPP[4] = {0};
   SPoint Index;
   NPP_CODE(
      cudaMemcpy(&NPP, m_NPPDst, sizeof(DstT) * 4, cudaMemcpyDeviceToHost);
      cudaMemcpy(&Index, m_IndxNPP, sizeof(Index), cudaMemcpyDeviceToHost);
      )

   double NPPd[4] = {NPP[0], NPP[1], NPP[2], NPP[3]};

   if (m_IndxIPP.X != 0 && m_IndxIPP.X != 0)
   {
      if (m_IndxIPP.X != Index.X)
         return false;
   
      if (m_IndxIPP.Y != Index.Y)
         return false;
   }

   return Compare(NPPd, m_DstIPP);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareCV(BenchReduceBase *)
{
#ifdef HAS_CV
   if (m_IndxIPP.X != m_IndxCV.x)
      return false;
   
   if (m_IndxIPP.Y != m_IndxCV.y)
      return false;

   double CV[4] = {m_DstCV[0], 0};

   return Compare(CV, m_DstIPP);
#else
   return false;
#endif
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::Compare(double * V1, DstT * V2)
{
   uint NbValues = m_ImgSrc.Channels;

   for (uint i = 0; i < NbValues; i++)
   {
      double diff = abs(V1[i] - V2[i]);
      double diffRatio = diff / abs(V2[i]);

      bool value = (V1[i] == V2[i]) || (diffRatio < CompareTolerance());
      if (!value)
         return false;
   }

   return true;
}
