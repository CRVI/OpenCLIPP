#ifndef __BENCH_MERGE_FROM_FILE_H__
#define __BENCH_MERGE_FROM_FILE_H__

#include <IVLArithmetic.h>
#include <ivlgpuCore.h>
#include <IVLFile.h>
#include <IVLImgUtility.h>

class CMergeBench8uFromFile
{
public:
   CMergeBench8uFromFile(ILModuleId cAppId)
   : m_cImgSrcC1(cAppId)
   , m_cImgSrcC2(cAppId)
   , m_cImgSrcC3(cAppId)
   , m_cImgDst(cAppId)
   , m_cImgUtility(cAppId)
   , m_cFile(cAppId)
   { ; }

   void Create(IVL_UINT uParam1, IVL_UINT uParam2);
   void Free();
   void RunCPU();
   void RunGPU();

   IVL_BOOL Compare();

private:
   IVLImgUtility m_cImgUtility;
   ILImageFile m_cFile;

   ILMainImage m_cImgSrcC1;
   ILMainImage m_cImgSrcC2;
   ILMainImage m_cImgSrcC3;
   ILMainImage m_cImgDst;

   IVL_UINT8* d_p8uSrcC1;
   IVL_UINT   m_uSrcStepC1;

   IVL_UINT8* d_p8uSrcC2;
   IVL_UINT   m_uSrcStepC2;

   IVL_UINT8* d_p8uSrcC3;
   IVL_UINT   m_uSrcStepC3;

   IVL_UINT8* d_p8uDst;
   IVL_UINT   m_uDstStep;
};
//-----------------------------------------------------------------------------------------------------------------------------
void CMergeBench8uFromFile::Create(IVL_UINT uParam1, IVL_UINT uParam2)
{
   IVL_UINT uWidth = 0;
   IVL_UINT uHeight = 0;

   static const char* IMG_SRC_C1 = "N:/ImageBank/689x439_8U_1C_Chateau_Frontenac_Blue.png";
   static const char* IMG_SRC_C2 = "N:/ImageBank/689x439_8U_1C_Chateau_Frontenac_Green.png";
   static const char* IMG_SRC_C3 = "N:/ImageBank/689x439_8U_1C_Chateau_Frontenac_Red.png";


   V( m_cFile.LoadImg(m_cImgSrcC1, IMG_SRC_C1));
   V( m_cFile.LoadImg(m_cImgSrcC2, IMG_SRC_C2));
   V( m_cFile.LoadImg(m_cImgSrcC3, IMG_SRC_C3));

   uWidth = m_cImgSrcC1.Width();
   uHeight = m_cImgSrcC1.Height();


   V(m_cImgDst.Create(uWidth,uHeight,3,U8));


   V( ivlgpuMalloc_8u_C1(d_p8uSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C1(d_p8uSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C1(d_p8uSrcC3, m_uSrcStepC3, uWidth, uHeight) );
   V( ivlgpuMalloc_8u_C3(d_p8uDst, m_uDstStep, uWidth, uHeight) );

   V( ivlgpuUpload_8u_C1(m_cImgSrcC1.GetPtr().pu8Data, m_cImgSrcC1.Step(), d_p8uSrcC1, m_uSrcStepC1, uWidth, uHeight) );
   V( ivlgpuUpload_8u_C1(m_cImgSrcC2.GetPtr().pu8Data, m_cImgSrcC2.Step(), d_p8uSrcC2, m_uSrcStepC2, uWidth, uHeight) );
   V( ivlgpuUpload_8u_C1(m_cImgSrcC3.GetPtr().pu8Data, m_cImgSrcC3.Step(), d_p8uSrcC3, m_uSrcStepC3, uWidth, uHeight) );

   V( ivlgpuUpload_8u_C3(m_cImgDst.GetPtr().pu8Data, m_cImgDst.Step(), d_p8uDst, m_uDstStep, uWidth, uHeight) );
}
//-----------------------------------------------------------------------------------------------------------------------------
void CMergeBench8uFromFile::Free()
{
   ivlgpuFree(d_p8uSrcC1);
   ivlgpuFree(d_p8uSrcC2);
   ivlgpuFree(d_p8uSrcC3);
   ivlgpuFree(d_p8uDst);

   m_cImgSrcC1.Free();
   m_cImgSrcC2.Free();
   m_cImgSrcC3.Free();
   m_cImgDst.Free();
}
//-----------------------------------------------------------------------------------------------------------------------------
void CMergeBench8uFromFile::RunCPU()
{
   V(m_cImgUtility.CopyGrayToColor(m_cImgSrcC1,m_cImgSrcC2,m_cImgSrcC3,m_cImgDst));
}
//-----------------------------------------------------------------------------------------------------------------------------
void CMergeBench8uFromFile::RunGPU()
{
   V( ivlgpuMerge_8u_C3(   d_p8uSrcC1, 
                           m_uSrcStepC1, 
                           d_p8uSrcC2, 
                           m_uSrcStepC2,
                           d_p8uSrcC3,
                           m_uSrcStepC3,
                           d_p8uDst,
                           m_uDstStep,
                           m_cImgSrcC1.Width(), 
                           m_cImgSrcC1.Height()));
}
//-----------------------------------------------------------------------------------------------------------------------------
IVL_BOOL CMergeBench8uFromFile::Compare()
{
   return CompareGpuToCpu(m_cImgDst, m_cImgSrcC1, d_p8uDst, m_uDstStep, SUCCESS_EPSILON);
}
//-----------------------------------------------------------------------------------------------------------------------------
#endif //__BENCH_MERGE_FROM_FILE_H__
