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
   , m_DstIPP(0)
   , m_DstCL(0)
   , m_CUDAWorkBuffer(nullptr)
   , m_NPPWorkBuffer(nullptr)
   { }

   void Create(uint Width, uint Height);
   void Free();

   bool CompareCL(BenchReduceBase * This);
   bool CompareNPP(BenchReduceBase * This);
   bool CompareCUDA(BenchReduceBase * This);
   bool CompareCV(BenchReduceBase * This);

   bool Compare(double V1, double V2);

   virtual float CompareTolerance() const { return SUCCESS_EPSILON; }

   DataType* Src() { return (DataType*) m_CUDASrc; }

   typedef DataType dataType;

protected:

   ocipProgram m_Program;

   DstT m_DstIPP;
   double m_DstCL;
   CV_CODE(Scalar m_DstCV;)

   DstT * m_NPPDst;

   unsigned char* m_CUDAWorkBuffer;
   unsigned char* m_NPPWorkBuffer;

   CUDA_CODE(CUDAPP(PageLockedArray) m_CUDADst;)

   friend class IBench1in0out;
};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
void BenchReduceBase<DataType, DstT>::Create(uint Width, uint Height)
{
   IBench1in0out::Create<DataType>(Width, Height);

   if (CLUsesBuffer())
      ocipPrepareImageBufferStatistics(&m_Program, m_CLBufferSrc);
   else
      ocipPrepareStatistics(&m_Program, m_CLSrc);

   CUDA_CODE(
      uint uBufferSize = 0;
      CUDAPP(MaxGetBufferSize<DataType>)(Width, Height, uBufferSize);
      CUDAPP(Malloc<DataType>)((DataType*&)m_CUDAWorkBuffer, uBufferSize);
      m_CUDADst.Create(sizeof(int));
      )

   NPP_CODE(
      int BufferSize = 0;
      nppiMaxGetBufferHostSize_32f_C1R(m_NPPRoi, &BufferSize);
      cudaMalloc((void**) &m_NPPWorkBuffer, BufferSize);
      cudaMalloc((void**) &m_NPPDst, sizeof(DstT));
      )

   if (std::is_same<float, DataType>::value)
   {
      // To prevent float values from overflowing, we divide the values to get them smaller

      if (USE_BUFFER)
      {
         // Create temporary image
         ocipBuffer TempImage = nullptr;
         ocipCreateImageBuffer(&TempImage, m_ImgSrc.ToSImage(), nullptr, CL_MEM_READ_WRITE);

         // Divide into temp
         ocipDivC_V(m_CLBufferSrc, TempImage, 1000000);

         // Copy temp to source image
         ocipCopy_V(TempImage, m_CLBufferSrc);

         // Read into host
         ocipReadImageBuffer(m_CLBufferSrc);

         ocipReleaseImageBuffer(TempImage);
      }
      else
      {
         // Create temporary image
         ocipImage TempImage = nullptr;
         ocipCreateImage(&TempImage, m_ImgSrc.ToSImage(), nullptr, CL_MEM_READ_WRITE);

         // Divide into temp
         ocipDivC(m_CLSrc, TempImage, 1000000);

         // Copy temp to source image
         ocipCopy(TempImage, m_CLSrc);

         // Read into host
         ocipReadImage(m_CLSrc);

         ocipReleaseImage(TempImage);
      }

      // Resend the image
      CUDA_CODE(
         CUDAPP(Upload<DataType>)((DataType*) m_ImgSrc.Data(), m_ImgSrc.Step,
            (DataType*) m_CUDASrc, m_CUDASrcStep, m_ImgSrc.Width, m_ImgSrc.Height);
         )

      NPP_CODE(
         cudaMemcpy2D(m_NPPSrc, m_NPPSrcStep, m_ImgSrc.Data(), m_ImgSrc.Step,
            m_ImgSrc.BytesWidth(), Height, cudaMemcpyHostToDevice);
         )
      
   }

}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
void BenchReduceBase<DataType, DstT>::Free()
{
   IBench1in0out::Free();

   ocipReleaseProgram(m_Program);

   CUDA_CODE(
      m_CUDADst.Free();
      CUDAPP(Free)(m_CUDAWorkBuffer);
      )

   NPP_CODE(
      cudaFree(m_NPPWorkBuffer);
      cudaFree(m_NPPDst);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareCL(BenchReduceBase *)
{
   return Compare(m_DstCL, m_DstIPP);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareNPP(BenchReduceBase *)
{
   DstT NPP = 0;
   NPP_CODE(cudaMemcpy(&NPP, m_NPPDst, sizeof(DstT), cudaMemcpyDeviceToHost);)

   return Compare(NPP, m_DstIPP);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareCV(BenchReduceBase *)
{
#ifdef HAS_CV
   return Compare(m_DstCV[0], m_DstIPP);
#else
   return false;
#endif
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::CompareCUDA(BenchReduceBase *)
{
   float DstCUDA = 0;

   CUDA_CODE(
      m_CUDADst.Download();
      DstT* h_pDst = reinterpret_cast<DstT*>(m_CUDADst.HostRef());
      DstCUDA = static_cast<float>(*h_pDst);
      )

   return Compare(DstCUDA, m_DstIPP);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType, typename DstT>
bool BenchReduceBase<DataType, DstT>::Compare(double V1, double V2)
{
   double diff = abs(V1 - V2);
   double diffRatio = diff / abs(V2);

   bool value = (V1 == V2) || (diffRatio < CompareTolerance());
   return value;
}
