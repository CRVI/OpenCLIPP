#ifndef __BENCH_MERGE_H__
#define __BENCH_MERGE_H__

#include <IVLArithmetic.h>
#include <ivlgpuCore.h>
#include <IVLFile.h>
#include <IVLImgUtility.h>

template<typename DataType> class CMergeBench;

typedef CMergeBench<IVL_INT8>   CMergeBenchS8; 
typedef CMergeBench<IVL_UINT8>  CMergeBenchU8; 
typedef CMergeBench<IVL_INT16>  CMergeBenchS16;
typedef CMergeBench<IVL_UINT16> CMergeBenchU16;
typedef CMergeBench<IVL_INT32>  CMergeBenchS32;
typedef CMergeBench<IVL_UINT32> CMergeBenchU32;
typedef CMergeBench<IVL_FLOAT>  CMergeBenchF32;

template<typename DataType>
class CMergeBench
{
public:
   CMergeBench(ILModuleId cAppId)
   : m_cImgSrcC1(cAppId)
   , m_cImgSrcC2(cAppId)
   , m_cImgSrcC3(cAppId)
   , m_cImgDst(cAppId)
   , m_cImgUtility(cAppId)
   { ; }

   void Create(IVL_UINT uWidth, IVL_UINT uHeight);
   void Free();
   void RunCPU();
   void RunGPU();

   IVL_BOOL Compare();

private:
   IVLImgUtility m_cImgUtility;

   ILMainImage m_cImgSrcC1;
   ILMainImage m_cImgSrcC2;
   ILMainImage m_cImgSrcC3;
   ILMainImage m_cImgDst;

   DataType* d_pSrcC1;
   IVL_UINT   m_uSrcStepC1;

   DataType* d_pSrcC2;
   IVL_UINT   m_uSrcStepC2;

   DataType* d_pSrcC3;
   IVL_UINT   m_uSrcStepC3;

   DataType* d_pDst;
   IVL_UINT   m_uDstStep;
};
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT8>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, S8) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, S8) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, S8) );
   V( m_cImgDst.Create(uWidth,uHeight,3,S8));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_8s_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_8s_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_8s_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_8s_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_8s_C1(m_cImgSrcC1.GetPtr().pi8Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_8s_C1(m_cImgSrcC2.GetPtr().pi8Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_8s_C1(m_cImgSrcC3.GetPtr().pi8Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_8s_C3(m_cImgDst.GetPtr().pi8Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT8>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, U8) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, U8) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, U8) );
   V( m_cImgDst.Create(uWidth,uHeight,3,U8));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_8u_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_8u_C1(m_cImgSrcC1.GetPtr().pu8Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_8u_C1(m_cImgSrcC2.GetPtr().pu8Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_8u_C1(m_cImgSrcC3.GetPtr().pu8Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_8u_C3(m_cImgDst.GetPtr().pu8Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}

//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT16>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, S16) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, S16) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, S16) );
   V( m_cImgDst.Create(uWidth,uHeight,3,S16));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_16s_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_16s_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_16s_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_16s_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_16s_C1(m_cImgSrcC1.GetPtr().pi16Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_16s_C1(m_cImgSrcC2.GetPtr().pi16Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_16s_C1(m_cImgSrcC3.GetPtr().pi16Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_16s_C3(m_cImgDst.GetPtr().pi16Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT16>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{

   IL_ERROR_STATUS ErrSts;

   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, U16) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, U16) );
   V( m_cImgDst.Create(uWidth,uHeight,3,U16));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );

   V( ivlgpuMalloc_16u_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_16u_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_16u_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_16u_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_16u_C1(m_cImgSrcC1.GetPtr().pu16Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_16u_C1(m_cImgSrcC2.GetPtr().pu16Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_16u_C1(m_cImgSrcC3.GetPtr().pu16Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_16u_C3(m_cImgDst.GetPtr().pu16Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT32>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, S32) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, S32) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, S32) );
   V( m_cImgDst.Create(uWidth,uHeight,3,S32));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_32s_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_32s_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_32s_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_32s_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_32s_C1(m_cImgSrcC1.GetPtr().pi32Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_32s_C1(m_cImgSrcC2.GetPtr().pi32Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_32s_C1(m_cImgSrcC3.GetPtr().pi32Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_32s_C3(m_cImgDst.GetPtr().pi32Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT32>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, U32) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, U32) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, U32) );
   V( m_cImgDst.Create(uWidth,uHeight,3,U32));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_32u_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_32u_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_32u_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_32u_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_32u_C1(m_cImgSrcC1.GetPtr().pu32Data, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_32u_C1(m_cImgSrcC2.GetPtr().pu32Data, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_32u_C1(m_cImgSrcC3.GetPtr().pu32Data, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_32u_C3(m_cImgDst.GetPtr().pu32Data, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_FLOAT>::Create(IVL_UINT uWidth, IVL_UINT uHeight)
{
   V( m_cImgSrcC1.Create(uWidth, uHeight, 1, F32) );
   V( m_cImgSrcC2.Create(uWidth, uHeight, 1, F32) );
   V( m_cImgSrcC3.Create(uWidth, uHeight, 1, F32) );
   V( m_cImgDst.Create(uWidth,uHeight,3,F32));

   V( FillRandomImg(m_cImgSrcC1) );
   V( FillRandomImg(m_cImgSrcC2) );
   V( FillRandomImg(m_cImgSrcC3) );


   V( ivlgpuMalloc_32f_C1(d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_32f_C1(d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_32f_C1(d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_32f_C3(d_pDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_32f_C1(m_cImgSrcC1.GetPtr().pfData, m_cImgSrcC1.Step(), d_pSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_32f_C1(m_cImgSrcC2.GetPtr().pfData, m_cImgSrcC2.Step(), d_pSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_32f_C1(m_cImgSrcC3.GetPtr().pfData, m_cImgSrcC3.Step(), d_pSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_32f_C3(m_cImgDst.GetPtr().pfData, m_cImgDst.Step(), d_pDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CMergeBench<DataType>::Free()
{
   ivlgpuFree(d_pSrcC1);
   ivlgpuFree(d_pSrcC2);
   ivlgpuFree(d_pSrcC3);
   ivlgpuFree(d_pDst);

   m_cImgSrcC1.Free();
   m_cImgSrcC2.Free();
   m_cImgSrcC3.Free();
   m_cImgDst.Free();
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
void CMergeBench<DataType>::RunCPU()
{
   V(m_cImgUtility.CopyGrayToColor(m_cImgSrcC1,m_cImgSrcC2,m_cImgSrcC3,m_cImgDst));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT8>::RunGPU()
{
   V( ivlgpuMerge_8s_C3(   d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT8>::RunGPU()
{
   V( ivlgpuMerge_8u_C3(   d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT16>::RunGPU()
{
   V( ivlgpuMerge_16s_C3(  d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT16>::RunGPU()
{
   V( ivlgpuMerge_16u_C3(  d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_INT32>::RunGPU()
{
   V( ivlgpuMerge_32s_C3(  d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_UINT32>::RunGPU()
{
   V( ivlgpuMerge_32u_C3(  d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<>
void CMergeBench<IVL_FLOAT>::RunGPU()
{
   V( ivlgpuMerge_32f_C3(  d_pSrcC1, 
                           m_uSrcStepC1, 
                           d_pSrcC2, 
                           m_uSrcStepC2,
                           d_pSrcC3,
                           m_uSrcStepC3,
                           d_pDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
template<typename DataType>
IVL_BOOL CMergeBench<DataType>::Compare()
{
   return CompareGpuToCpu(m_cImgDst, d_pDst, m_uDstStep, SUCCESS_EPSILON);
}
//-----------------------------------------------------------------------------------------------------------------------------
#endif //__BENCH_MERGE_H__
