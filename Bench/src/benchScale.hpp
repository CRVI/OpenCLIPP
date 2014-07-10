////////////////////////////////////////////////////////////////////////////////
//! @file	: benchScale.hpp
//! @date   : Jul 2013
//!
//! @brief  : Benchmark class for image scaling (value scaling based on data type)
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

template<typename SrcType, typename DstType> class ScaleBench;

typedef ScaleBench<unsigned char, unsigned short>  ScaleBenchU8;
//typedef ScaleBench<unsigned short, uint> ScaleBenchU16;
//typedef ScaleBench<unsigned short, int>  ScaleBenchS16;
//typedef ScaleBench<int, uint>  ScaleBenchS32;


template<typename SrcType, typename DstType>
class ScaleBench : public IBench1in1out
{
public:
   ScaleBench()
   { }

   void Create(uint Width, uint Height);
   void RunIPP();
   void RunCL();
   void RunNPP();

   bool HasCVTest() const { return false; }

};
//-----------------------------------------------------------------------------------------------------------------------------
template<typename SrcType, typename DstType>
void ScaleBench<SrcType, DstType>::Create(uint Width, uint Height)
{
   IBench1in1out::Create<SrcType, DstType>(Width, Height);
}

//-----------------------------------------------------------------------------------------------------------------------------
template<typename SrcType, typename DstType>
void ScaleBench<SrcType, DstType>::RunCL()
{
   ocipScale(m_CLSrc, m_CLDst);
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ScaleBench<unsigned char, unsigned short>::RunNPP()
{
   NPP_CODE(nppiScale_8u16u_C1R((Npp8u*) m_NPPSrc, m_NPPSrcStep, (Npp16u*) m_NPPDst, m_NPPDstStep, m_NPPRoi);)
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void ScaleBench<unsigned char, unsigned short>::RunIPP()
{
   IPP_CODE(ippiScale_8u16u_C1R(m_ImgSrc.Data(), m_ImgSrc.Step, (Ipp16u*) m_ImgDstIPP.Data(), m_ImgDstIPP.Step, m_IPPRoi);)
}
