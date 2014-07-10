////////////////////////////////////////////////////////////////////////////////
//! @file	: benchImageProximity.hpp
//! @date   : Mar 2014
//!
//! @brief  : Creates a benchmark class for Image Proximity operations
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

#define CLASS_NAME CONCATENATE(BENCH_NAME, Bench)
template<typename DataType> class CLASS_NAME;

#ifndef IPP_NAME
#define IPP_NAME BENCH_NAME
#endif

template<typename DataType>
class CLASS_NAME : public IBench1in1out
{
public:

#ifdef IMG_PROX_FFT
   const static int TemplateSize = 200;
#else
   const static int TemplateSize = 16;
#endif

   CLASS_NAME()
   :  m_Program(nullptr),
      m_NPPTemplate(nullptr)
   { }

   void RunIPP();
   void RunNPP();
   void RunCL();

   bool HasCVTest()   const { return false; }

   void Create(uint Width, uint Height);
   void Free();

   float CompareTolerance() const { return 0.01f; }
   bool CompareTolRelative() const { return true; }

protected:

   std::unique_ptr<CImageROI> m_ImgTemplate;

   ocipImage m_CLBufTemplate;

   void * m_NPPTemplate;
   int m_NPPTemplateStep;

   ocipProgram m_Program;

   IPP_CODE( IppiSize m_IPPTemplateSize; )

   NPP_CODE( NppiSize m_NPPTemplateSize; )
};

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<DataType, float>(Width, Height);

   // CL
   m_ImgTemplate = std::unique_ptr<CImageROI>(new CImageROI(m_ImgSrc, 10, 10,
      min(TemplateSize, int(m_ImgSrc.Width) - 10), min(TemplateSize, int(m_ImgSrc.Height) - 10)));

   ocipCreateImage(&m_CLBufTemplate, m_ImgTemplate->ToSImage(), m_ImgTemplate->Data(), CL_MEM_READ_WRITE);

#ifdef IMG_PROX_FFT
   ocipPrepareImageProximityFFT(&m_Program, m_CLBufferSrc, m_CLBufTemplate);
#endif

   // IPP
   IPP_CODE(
      m_IPPTemplateSize.width = m_ImgTemplate->Width;
      m_IPPTemplateSize.height = m_ImgTemplate->Height;
      )

   // NPP
   NPP_CODE(
      m_NPPTemplateSize.width = m_ImgTemplate->Width;
      m_NPPTemplateSize.height = m_ImgTemplate->Height;

      m_NPPTemplate = NPP_Malloc<DataType>(m_ImgTemplate->Width, m_ImgTemplate->Height,
         m_NPPTemplateStep, 1);

      cudaMemcpy2D(m_NPPTemplate, m_NPPTemplateStep, m_ImgTemplate->Data(), m_ImgTemplate->Step,
         m_ImgTemplate->BytesWidth(), m_ImgTemplate->Height, cudaMemcpyHostToDevice);
      )
  
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::Free()
{
   IBench1in1out::Free();

   ocipReleaseImage(m_CLBufTemplate);

   ocipReleaseProgram(m_Program);
   m_Program = nullptr;

   NPP_CODE(nppiFree(m_NPPTemplate);)
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunIPP()
{
   IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _8u32f_C1R)( this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                              this->m_IPPRoi, this->m_ImgTemplate->Data(), this->m_ImgTemplate->Step, this->m_IPPTemplateSize, 
                                                              (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunIPP()
{
    IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _16u32f_C1R)( (Ipp16u*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                               this->m_IPPRoi, (Ipp16u*)this->m_ImgTemplate->Data(), this->m_ImgTemplate->Step, this->m_IPPTemplateSize, 
                                                               (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunIPP()
{
   IPP_CODE(
        CONCATENATE(CONCATENATE(ippi, IPP_NAME), _32f_C1R)( (Ipp32f*)this->m_ImgSrc.Data(), this->m_ImgSrc.Step, 
                                                            this->m_IPPRoi, (Ipp32f*)this->m_ImgTemplate->Data(), this->m_ImgTemplate->Step, this->m_IPPTemplateSize, 
                                                            (Ipp32f*)this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned char>::RunNPP()
{
   NPP_CODE(
        CONCATENATE(CONCATENATE(nppi, IPP_NAME), _8u32f_C1R)( (Npp8u*) this->m_NPPSrc, this->m_NPPSrcStep, 
                                                              this->m_NPPRoi, (Npp8u*) this->m_NPPTemplate, this->m_NPPTemplateStep, this->m_NPPTemplateSize, 
                                                              (Npp32f*) this->m_NPPDst, this->m_NPPDstStep);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<unsigned short>::RunNPP()
{
    NPP_CODE(
        CONCATENATE(CONCATENATE(nppi, IPP_NAME), _16u32f_C1R)( (Npp16u*) this->m_NPPSrc, this->m_NPPSrcStep, 
                                                               this->m_NPPRoi, (Npp16u*) this->m_NPPTemplate, this->m_NPPTemplateStep, this->m_NPPTemplateSize, 
                                                               (Npp32f*) this->m_NPPDst, this->m_NPPDstStep);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CLASS_NAME<float>::RunNPP()
{
   NPP_CODE(
        CONCATENATE(CONCATENATE(nppi, IPP_NAME), _32f_C1R)( (Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep, 
                                                            this->m_NPPRoi, (Npp32f*) this->m_NPPTemplate, this->m_NPPTemplateStep, this->m_NPPTemplateSize, 
                                                            (Npp32f*) this->m_NPPDst, this->m_NPPDstStep);
      )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CLASS_NAME<DataType>::RunCL()
{
#ifdef IMG_PROX_FFT
   CONCATENATE(ocip, BENCH_NAME)(m_Program, m_CLBufferSrc, m_CLBufTemplate, m_CLBufferDst);
#else
   CONCATENATE(ocip, BENCH_NAME)(m_CLBufferSrc, m_CLBufTemplate, m_CLBufferDst);
#endif
}

#undef CLASS_NAME
#undef BENCH_NAME
#undef IPP_NAME
