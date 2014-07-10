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


template<typename DataType, int mask_size = 1, int channels = 1>
class FilterBenchBase : public BenchUnaryBase<DataType>
{
public:
   void Create(uint Width, uint Height);

   SSize CompareSize() const { return m_MaskSize; }
   SPoint CompareAnchor() const { return m_MaskAnchor; }
   float CompareTolerance() const { return FilterTolerance<DataType>(); }

   bool HasCVTest() const { return false; }   // API of filters in OpenCV OCL is too different to be easily added here

   SSize m_MaskSize;
   SPoint m_MaskAnchor;
};

#define FILTER_CLASS_NAME(Name, width, nb_channels) CONCATENATE(CONCATENATE(Name, CONCATENATE(width, CONCATENATE(_, nb_channels))), Bench)

#define DECLARE_FILTER_BENCH(Name, width, nb_channels) \
class FILTER_CLASS_NAME(Name, width, nb_channels) : public FilterBenchBase<FILTER_TYPE, width / 2, nb_channels>\
{\
public:\
   void RunCL()\
   {\
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

#define FILTER_BENCH(Name, width)  DECLARE_FILTER_BENCH(Name, width, 1)
#define FILTER_BENCH4(Name, width) DECLARE_FILTER_BENCH(Name, width, 4)

template<typename DataType, int mask_size, int channels>
void FilterBenchBase<DataType, mask_size, channels>::Create(uint Width, uint Height)
{
   BenchUnaryBase<DataType>::Create(Width, Height, channels);

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

#undef  FILTER_IPP_MOD
#define FILTER_IPP_MOD _8u_C4R

FILTER_BENCH4(Sharpen, 3)
FILTER_BENCH4(SobelVert, 3)
FILTER_BENCH4(SobelHoriz, 3)

// F32 images (not supported with U8 in IPP)
#undef FILTER_TYPE
#undef FILTER_IPP_MOD
#define FILTER_TYPE float
#define FILTER_IPP_MOD _32f_C1R

FILTER_BENCH(ScharrVert, 3)
FILTER_BENCH(ScharrHoriz, 3)
FILTER_BENCH(PrewittVert, 3)
FILTER_BENCH(PrewittHoriz, 3)

#undef FILTER_IPP_MOD
#define FILTER_IPP_MOD _32f_C4R

//FILTER_BENCH(ScharrVert, 3)    // 4C Scharr not supported in IPP
//FILTER_BENCH(ScharrHoriz, 3)
FILTER_BENCH4(PrewittVert, 3)
FILTER_BENCH4(PrewittHoriz, 3)


// These primitives need to have a mask

#undef FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD _32f_C1R
#define FILTERS_IPP_MASK , ippMskSize3x3
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_3_X_3

FILTER_BENCH(Laplace, 3)
FILTER_BENCH(SobelCross, 3)
FILTER_BENCH(Gauss, 3)

#undef  FILTER_IPP_MOD
#define FILTER_IPP_MOD _32f_C4R

FILTER_BENCH4(Laplace, 3)
//FILTER_BENCH(SobelCross, 3) // 4C SobelCross not supported in IPP
FILTER_BENCH4(Gauss, 3)


#undef  FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD _32f_C1R
#define FILTERS_IPP_MASK , ippMskSize5x5
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_5_X_5

FILTER_BENCH(Laplace, 5)
FILTER_BENCH(SobelCross, 5)
FILTER_BENCH(Gauss, 5)

#undef  FILTER_IPP_MOD
#define FILTER_IPP_MOD _32f_C4R

FILTER_BENCH4(Laplace, 5)
//FILTER_BENCH(SobelCross, 5) // 4C SobelCross not supported in IPP
FILTER_BENCH4(Gauss, 5)

#undef FILTER_IPP_MOD
#define FILTER_IPP_MOD Mask_32f_C1R

FILTER_BENCH(SobelVert, 5)
FILTER_BENCH(SobelHoriz, 5)

//FILTER_BENCH4(SobelVert, 5) // 32f 5x5 4C Sobel not supported in IPP
//FILTER_BENCH4(SobelVert, 5)


// TODO : Make a version that tests filters with Anchor and MaskSize like Median and Box

// These are not supported as a single operation in IPP and NPP
// So they are made in multi-step

template<typename DataType, int width, int channels>
class AdvancedFilter : public FilterBenchBase<FILTER_TYPE, width, channels>
{
public:
   void Create(uint Width, uint Height)
   {
      FilterBenchBase<FILTER_TYPE, width, channels>::Create(Width, Height);

      m_IPPTmpV.Create<DataType>(Width, Height, channels);

      // NPP
      NPP_CODE(
         m_NPPTmpV = NPP_Malloc<DataType>(Width, Height, m_NPPTmpVStep, channels);
         )
   }

   void Free()
   {
      FilterBenchBase<FILTER_TYPE, width, channels>::Free();

      NPP_CODE(nppiFree(m_NPPTmpV);)
   }

protected:

   CSimpleImage m_IPPTmpV;

   void * m_NPPTmpV;
   int m_NPPTmpVStep;
};

#undef DECLARE_FILTER_BENCH
#define DECLARE_FILTER_BENCH(Name, width, nb_channels) \
class FILTER_CLASS_NAME(Name, width, nb_channels) : public AdvancedFilter<FILTER_TYPE, width / 2, nb_channels>\
{\
public:\
   void RunCL()\
   {\
      CONCATENATE(ocip, Name) (m_CLSrc, m_CLDst, width);\
   }\
   void RunIPP()\
   {\
      IPP_CODE( \
         CONCATENATE(CONCATENATE(ippiFilter, CONCATENATE(Name, Vert)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_ImgSrc.Data(width / 2, width / 2), m_ImgSrc.Step, (FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, m_IPPRoi FILTERS_IPP_MASK);\
         CONCATENATE(CONCATENATE(ippiFilter, CONCATENATE(Name, Horiz)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_ImgSrc.Data(width / 2, width / 2), m_ImgSrc.Step, (FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi FILTERS_IPP_MASK);\
         CONCATENATE(ippiSqr, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, m_IPPRoi);\
         CONCATENATE(ippiSqr, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi);\
         CONCATENATE(ippiAdd, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_IPPTmpV.Data(width / 2, width / 2), m_IPPTmpV.Step, (FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi);\
         CONCATENATE(ippiSqrt, FILTERS_MATH_SUFFIX(nb_channels)) ((FILTER_TYPE*) m_ImgDstIPP.Data(width / 2, width / 2), m_ImgDstIPP.Step, m_IPPRoi);\
      )\
   }\
   void RunNPP()\
   {\
      NPP_CODE(\
         CONCATENATE(CONCATENATE(nppiFilter, CONCATENATE(Name, Vert)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_NPPSrc, m_NPPSrcStep, (FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, m_NPPRoi FILTERS_NPP_MASK);\
         CONCATENATE(CONCATENATE(nppiFilter, CONCATENATE(Name, Horiz)), FILTER_IPP_MOD)\
            ((FILTER_TYPE*) m_NPPSrc, m_NPPSrcStep, (FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi FILTERS_NPP_MASK);\
         CONCATENATE(nppiSqr, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, m_NPPRoi);\
         CONCATENATE(nppiSqr, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi);\
         CONCATENATE(nppiAdd, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_NPPTmpV, m_NPPTmpVStep, (FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi);\
         CONCATENATE(nppiSqrt, FILTERS_MATH_SUFFIX(nb_channels))\
            ((FILTER_TYPE*) m_NPPDst, m_NPPDstStep, m_NPPRoi);\
         )\
   }\
};

#define FILTERS_MATH_SUFFIX(nb_channels) CONCATENATE(_32f_C, CONCATENATE(nb_channels, IR))

#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTERS_IPP_MASK , ippMskSize3x3
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_3_X_3

FILTER_BENCH(Sobel, 3)

#undef FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD _32f_C4R
#define FILTERS_IPP_MASK
#define FILTERS_NPP_MASK

FILTER_BENCH4(Sobel, 3)

#undef FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD Mask_32f_C1R
#define FILTERS_IPP_MASK , ippMskSize5x5
#define FILTERS_NPP_MASK , NPP_MASK_SIZE_5_X_5

FILTER_BENCH(Sobel, 5)
//FILTER_BENCH4(Sobel, 5)  // 32f 5x5 4C Sobel not supported in IPP

#undef FILTER_IPP_MOD
#undef FILTERS_IPP_MASK
#undef FILTERS_NPP_MASK
#define FILTER_IPP_MOD _32f_C1R
#define FILTERS_IPP_MASK
#define FILTERS_NPP_MASK

FILTER_BENCH(Prewitt, 3)
FILTER_BENCH(Scharr, 3)

#undef FILTER_IPP_MOD
#define FILTER_IPP_MOD _32f_C4R

FILTER_BENCH4(Prewitt, 3)
//FILTER_BENCH(Scharr, 3)  // 4C Scharr not supported in IPP
