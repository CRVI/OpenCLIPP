////////////////////////////////////////////////////////////////////////////////
//! @file	: benchConvert.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class datatype conversions
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

using namespace std;

template<typename SrcType, typename DstType> class ConvertBench;

typedef ConvertBench<unsigned char, unsigned short>   ConvertBenchU8;
typedef ConvertBench<unsigned short, float>           ConvertBenchU16;
typedef ConvertBench<short, float>                    ConvertBenchS16;
typedef ConvertBench<int, float>                      ConvertBenchS32;
typedef ConvertBench<float, unsigned short>           ConvertBenchF32;
typedef ConvertBench<float, unsigned char>            ConvertBenchFU8;

template<typename SrcType, typename DstType>
class ConvertBench : public IBench1in1out
{
public:
   ConvertBench()
   :  IBench1in1out(USE_BUFFER)
   { }

   void Create(uint Width, uint Height);
   void RunIPP();
   void RunCL();
   void RunCV();
   void RunNPP();

   bool HasCVTest() const { return false; }  // OpenCV OCL fails when trying to convert images

   float CompareTolerance() const { return 1; }   // Increased tolerance to accept minor rounding errors

};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename SrcType, typename DstType>
void ConvertBench<SrcType, DstType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<SrcType, DstType>(Width, Height);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename SrcType, typename DstType>
void ConvertBench<SrcType, DstType>::RunCL()
{
   if (m_UsesBuffer)
      ocipConvert_V(m_CLBufferSrc, m_CLBufferDst);
   else
      ocipConvert(m_CLSrc, m_CLDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename SrcType, typename DstType>
void ConvertBench<SrcType, DstType>::RunCV()
{
   CV_CODE( 
      try
      {
         m_CVSrc.convertTo(m_CVDst, m_CVDst.type());
      }
      catch (cv::Exception e)
      { } )
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<unsigned char, unsigned short>::RunNPP()
{
   NPP_CODE(nppiConvert_8u16u_C1R((Npp8u*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<unsigned short, float>::RunNPP()
{
   NPP_CODE(nppiConvert_16u32f_C1R((Npp16u*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<short, float>::RunNPP()
{
   NPP_CODE(nppiConvert_16s32f_C1R((Npp16s*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<int, float>::RunNPP()
{
   NPP_CODE(nppiConvert_32s32f_C1R((Npp32s*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp32f*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<float, unsigned short>::RunNPP()
{
   NPP_CODE(nppiConvert_32f16u_C1R((Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp16u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, NPP_RND_NEAR);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<float, unsigned char>::RunNPP()
{
   NPP_CODE(nppiConvert_32f8u_C1R((Npp32f*) this->m_NPPSrc, this->m_NPPSrcStep, (Npp8u*) this->m_NPPDst, this->m_NPPDstStep, this->m_NPPRoi, NPP_RND_NEAR);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<unsigned char, unsigned short>::RunIPP()
{
   IPP_CODE(ippiConvert_8u16u_C1R(this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<unsigned short, float>::RunIPP()
{
   IPP_CODE(ippiConvert_16u32f_C1R((Ipp16u*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<short, float>::RunIPP()
{
   IPP_CODE(ippiConvert_16s32f_C1R((Ipp16s*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<int, float>::RunIPP()
{
   IPP_CODE(ippiConvert_32s32f_C1R((Ipp32s*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp32f*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<float, unsigned short>::RunIPP()
{
   IPP_CODE(ippiConvert_32f16u_C1R((Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp16u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, ippRndNear);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ConvertBench<float, unsigned char>::RunIPP()
{
   IPP_CODE(ippiConvert_32f8u_C1R((Ipp32f*) this->m_ImgSrc.Data(), this->m_ImgSrc.Step, (Ipp8u*) this->m_ImgDstIPP.Data(), this->m_ImgDstIPP.Step, this->m_IPPRoi, ippRndNear);)
}
