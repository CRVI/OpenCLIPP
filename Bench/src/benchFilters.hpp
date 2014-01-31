////////////////////////////////////////////////////////////////////////////////
//! @file	: benchFilters.hpp
//! @date   : Jan 2014
//!
//! @brief  : Benchmark classes for Filters
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


const static bool FiltersUseBuffer = true;

template<typename DataType>
float FilterTolerance()
{
   return 1;   // Allow slight variations
}

template<>
float FilterTolerance<float>()
{
   return 1.f/100;   // Allow slight variations
}


template<typename DataType, int mask_size = 1>
class FilterBenchBase : public BenchUnaryBase<DataType, FiltersUseBuffer>
{
public:
   void Create(uint Width, uint Height);

   SSize CompareSize() const { return m_MaskSize; }
   SPoint CompareAnchor() const { return m_MaskAnchor; }
   float CompareTolerance() const { return FilterTolerance<DataType>(); }

   SSize m_MaskSize;
   SPoint m_MaskAnchor;
};

#define FILTER_BENCH(Name, width) \
class CONCATENATE(CONCATENATE(Name, width), Bench) : public FilterBenchBase<FILTER_TYPE, width / 2>\
{\
public:\
   void RunCL()\
   {\
      if (CLUsesBuffer())\
         CONCATENATE(CONCATENATE(ocip, Name), _V) (m_CLBufferSrc, m_CLBufferDst, width);\
      else\
         CONCATENATE(ocip, Name) (m_CLSrc, m_CLDst, width);\
   }\
   void RunIPP()\
   {\
      IPP_CODE( \
         CONCATENATE(CONCATENATE(ippiFilter, Name), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_ImgSrc.Data(width / 2, width / 2), m_ImgSrc.Step, (FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi FILTERS_IPP_MASK);\
      )\
   }\
   void RunNPP()\
   {\
      NPP_CODE(\
         CONCATENATE(CONCATENATE(nppiFilter, Name), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_NPPSrc, m_NPPSrcStep, (FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi FILTERS_NPP_MASK);\
         )\
   }\
};

template<typename DataType, int mask_size>
void FilterBenchBase<DataType, mask_size>::Create(uint Width, uint Height)
{
   BenchUnaryBase<DataType, FiltersUseBuffer>::Create(Width, Height);

   IPP_CODE(
      m_IPPRoi.width -= mask_size * 2;
      m_IPPRoi.height -= mask_size * 2;
      )

   m_MaskSize = SSize(mask_size * 2 + 1, mask_size * 2 + 1);
   m_MaskAnchor = SPoint(mask_size, mask_size);
}

// U8 images
#define FILTER_TYPE unsigned char
#define FILTER_IPP_MOD _8u_C1R
#define FILTERS_IPP_MASK
#define FILTERS_NPP_MASK

FILTER_BENCH(Sharpen, 3)
FILTER_BENCH(SobelVert, 3)
FILTER_BENCH(SobelHoriz, 3)

//FILTER_BENCH(Smooth, 3)  // Called Box in IPP and needs a mask and anchor
//FILTER_BENCH(Smooth, 5)
//FILTER_BENCH(Smooth, 7)

//FILTER_BENCH(Hipass, 3)  // Not supported by NPP
//FILTER_BENCH(Hipass, 5)  // Not supported by NPP

// F32 images (not supported with U8 in IPP)
#undef FILTER_TYPE
#undef FILTER_IPP_MOD
#define FILTER_TYPE float
#define FILTER_IPP_MOD _32f_C1R

FILTER_BENCH(ScharrVert, 3)
FILTER_BENCH(ScharrHoriz, 3)
FILTER_BENCH(PrewittVert, 3)
FILTER_BENCH(PrewittHoriz, 3)

// These primitives need to have a mask

#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTERS_IPP_MASK , ippMskSize3x3
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_3_X_3

FILTER_BENCH(Laplace, 3)
FILTER_BENCH(SobelCross, 3)
FILTER_BENCH(Gauss, 3)

#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTERS_IPP_MASK , ippMskSize5x5
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_5_X_5

FILTER_BENCH(Laplace, 5)
FILTER_BENCH(SobelCross, 5)
FILTER_BENCH(Gauss, 5)

#undef FILTER_IPP_MOD
#define FILTER_IPP_MOD Mask_32f_C1R

FILTER_BENCH(SobelVert, 5)
FILTER_BENCH(SobelHoriz, 5)

// TODO : Make a version that tests filters with Anchor and MaskSize like Median and Box

// These are not supported as a single operation in IPP and NPP
// So they are made in multi-step

template<typename DataType, int width>
class AdvancedFilter : public FilterBenchBase<FILTER_TYPE, width>
{
public:
   void Create(uint Width, uint Height)
   {
      FilterBenchBase<FILTER_TYPE, width>::Create(Width, Height);

      m_IPPTmpV.Create<DataType>(Width, Height);
      m_IPPTmpH.Create<DataType>(Width, Height);

      // NPP
      NPP_CODE(
         m_NPPTmpV = NPP_Malloc<sizeof(DataType)>(Width, Height, m_NPPTmpVStep);
         m_NPPTmpH = NPP_Malloc<sizeof(DataType)>(Width, Height, m_NPPTmpHStep);
         )

      // CUDA
      /*CUDA_CODE(
         CUDAPP(Malloc<DataType>)((DataType*&) m_CUDATmpV, m_CUDATmpVStep, Width, Height);
         CUDAPP(Malloc<DataType>)((DataType*&) m_CUDATmpH, m_CUDATmpHStep, Width, Height);
         )*/
   }

   void Free()
   {
      FilterBenchBase<FILTER_TYPE, width>::Free();

      NPP_CODE(nppiFree(m_NPPTmpV);)
      NPP_CODE(nppiFree(m_NPPTmpH);)

      //CUDA_CODE(CUDAPP(Free)(m_CUDASrc);)
   }

protected:

   CSimpleImage m_IPPTmpV;
   CSimpleImage m_IPPTmpH;

   void* m_CUDATmpV;
   uint  m_CUDATmpVStep;
   void* m_CUDATmpH;
   uint  m_CUDATmpHStep;

   void * m_NPPTmpV;
   int m_NPPTmpVStep;
   void * m_NPPTmpH;
   int m_NPPTmpHStep;
};

#undef FILTER_BENCH
#define FILTER_BENCH(Name, width) \
class CONCATENATE(CONCATENATE(Name, width), Bench) : public AdvancedFilter<FILTER_TYPE, width / 2>\
{\
public:\
   void RunCL()\
   {\
      if (CLUsesBuffer())\
         CONCATENATE(CONCATENATE(ocip, Name), _V) (m_CLBufferSrc, m_CLBufferDst, width);\
      else\
         CONCATENATE(ocip, Name) (m_CLSrc, m_CLDst, width);\
   }\
   void RunIPP()\
   {\
      IPP_CODE( \
         CONCATENATE(CONCATENATE(ippiFilter, CONCATENATE(Name, Vert)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_ImgSrc.Data(width / 2, width / 2), m_ImgSrc.Step, (FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, m_IPPRoi FILTERS_IPP_MASK);\
         CONCATENATE(CONCATENATE(ippiFilter, CONCATENATE(Name, Horiz)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_ImgSrc.Data(width / 2, width / 2), m_ImgSrc.Step, (FILTER_TYPE*) m_IPPTmpH.Data(width / 2, width / 2), m_IPPTmpH.Step, m_IPPRoi FILTERS_IPP_MASK);\
         ippiSqr_32f_C1IR((FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, m_IPPRoi);\
         ippiSqr_32f_C1IR((FILTER_TYPE*) m_IPPTmpH.Data(width / 2, width / 2), m_IPPTmpH.Step, m_IPPRoi);\
         ippiAdd_32f_C1IR((FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, (FILTER_TYPE*) m_IPPTmpH.Data(width / 2, width / 2), m_IPPTmpH.Step, m_IPPRoi);\
         ippiSqrt_32f_C1R((FILTER_TYPE*) m_IPPTmpH.Data(width / 2, width / 2), m_IPPTmpH.Step, (FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi);\
      )\
   }\
   void RunNPP()\
   {\
      NPP_CODE(\
         CONCATENATE(CONCATENATE(nppiFilter, CONCATENATE(Name, Vert)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_NPPSrc, m_NPPSrcStep, (FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, m_NPPRoi FILTERS_NPP_MASK);\
         CONCATENATE(CONCATENATE(nppiFilter, CONCATENATE(Name, Horiz)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_NPPSrc, m_NPPSrcStep, (FILTER_TYPE*) m_NPPTmpH, m_NPPTmpHStep, m_NPPRoi FILTERS_NPP_MASK);\
         nppiSqr_32f_C1IR((FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, m_NPPRoi);\
         nppiSqr_32f_C1IR((FILTER_TYPE*) m_NPPTmpH, m_NPPTmpHStep, m_NPPRoi);\
         nppiAdd_32f_C1IR((FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, (FILTER_TYPE*) m_NPPTmpH, m_NPPTmpHStep, m_NPPRoi);\
         nppiSqrt_32f_C1R((FILTER_TYPE*) m_NPPTmpH, m_NPPTmpHStep, (FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi);\
         )\
   }\
};

#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTERS_IPP_MASK , ippMskSize3x3
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_3_X_3

FILTER_BENCH(Sobel, 3)

#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTERS_IPP_MASK , ippMskSize5x5
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_5_X_5

FILTER_BENCH(Sobel, 5)

#undef FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD _32f_C1R
#define FILTERS_IPP_MASK
#define FILTERS_NPP_MASK

FILTER_BENCH(Prewitt, 3)
FILTER_BENCH(Scharr, 3)
