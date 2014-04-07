////////////////////////////////////////////////////////////////////////////////
//! @file	: Compare.h
//! @date   : Jul 2013
//!
//! @brief  : Compares two images
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

bool AbsDiff(const CSimpleImage& Source1, const CSimpleImage& Source2, CSimpleImage& Dest)
{
   if (Source1.Type != Source2.Type)
      return false;

   if (Source1.Type != Dest.Type)
      return false;

   if (Source1.Channels != Source2.Channels)
      return false;

   IppiSize Roi = {Dest.Width * Source1.Channels, Dest.Height};

   switch (Source1.Type)
   {
   case SImage::U8:
      ippiAbsDiff_8u_C1R(Source1.Data(), Source1.Step,
         Source2.Data(), Source2.Step,
         Dest.Data(), Dest.Step, Roi);
      break;
   case SImage::U16:
      ippiAbsDiff_16u_C1R((Ipp16u*) Source1.Data(), Source1.Step,
         (Ipp16u*) Source2.Data(), Source2.Step,
         (Ipp16u*) Dest.Data(), Dest.Step, Roi);
      break;
   case SImage::F32:
      ippiAbsDiff_32f_C1R((Ipp32f*) Source1.Data(), Source1.Step,
         (Ipp32f*) Source2.Data(), Source2.Step,
         (Ipp32f*) Dest.Data(), Dest.Step, Roi);
      break;
   case SImage::F64:
      // IPP does not support AbsDiff for 64b
      // Use OpenCLIPP instead
      {
         ocipBuffer buf1 = nullptr, buf2 = nullptr, dest = nullptr;
         ocipCreateImageBuffer(&buf1, Source1, (void *) Source1.Data(), CL_MEM_READ_WRITE);
         ocipCreateImageBuffer(&buf2, Source2, (void *) Source2.Data(), CL_MEM_READ_WRITE);
         ocipCreateImageBuffer(&dest, Dest, (void *) Dest.Data(), CL_MEM_READ_WRITE);

         ocipAbsDiff_V(buf1, buf2, dest);

         ocipReadImageBuffer(dest);

         ocipReleaseImageBuffer(buf1);
         ocipReleaseImageBuffer(buf2);
         ocipReleaseImageBuffer(dest);
      }
      break;
   default:
      return false;
   }

   return true;
}

float FindMax(const CSimpleImage& Source, SPoint Offset, SSize RoiSize, SPoint& Index)
{
   IppiSize Roi = {RoiSize.Width * Source.Channels, RoiSize.Height};
   Ipp8u V8 = 0;
   Ipp16u V16 = 0;
   Ipp32f VF = 0;

   const unsigned char * Data = Source.Data(Offset.X, Offset.Y);

   int PosX = 0, PosY = 0;
   float Result;

   switch (Source.Type)
   {
   case SImage::U8:
      ippiMaxIndx_8u_C1R(Data, Source.Step, Roi, &V8, &PosX, &PosY);
      Result = V8;
      break;
   case SImage::U16:
      ippiMaxIndx_16u_C1R((Ipp16u*) Data, Source.Step, Roi, &V16, &PosX, &PosY);
      Result = V16;
      break;
   case SImage::F32:
      ippiMaxIndx_32f_C1R((Ipp32f*) Data, Source.Step, Roi, &VF, &PosX, &PosY);
      Result = VF;
      break;
   case SImage::F64:
      // MaxIndx F64 is not supported in IPP
      {
         double res = 0;
         ocipBuffer buf = nullptr;
         ocipProgram program = nullptr;
         ocipCreateImageBuffer(&buf, Source, (void *) Source.Data(), CL_MEM_READ_WRITE);
         ocipPrepareImageBufferStatistics(&program, buf);
         ocipMaxIndx_V(program, buf, &res, &PosX, &PosY);
         Result = (float) res;

         ocipReleaseImageBuffer(buf);
         ocipReleaseProgram(program);
      }
      break;
   default:
      Result = 0;
   }

   Index.X = PosX / Source.Channels;
   Index.Y = PosY;

   return Result;
}

template<class T>
static inline bool CompareImages(const CSimpleImage& Img1,
                                 const CSimpleImage& Img2,
                                 const CSimpleImage& ImgSrc,
                                 const T& CompareInfo)
{
   float Tolerance = CompareInfo.CompareTolerance();
   SSize MaskSize = CompareInfo.CompareSize();
   SPoint MaskAnchor = CompareInfo.CompareAnchor();
   bool RelativeTolerance = CompareInfo.CompareTolRelative();

   CSimpleImage ImgAbsDiff(Img1.ToSImage());

   bool Success = AbsDiff(Img1, Img2, ImgAbsDiff);

   if (!Success)
      return false;

   SSize RoiSize(Img1.Width - MaskSize.Width + 1,
                 Img1.Height - MaskSize.Height + 1);

   SPoint Index;

   float Max = FindMax(ImgAbsDiff, MaskAnchor, RoiSize, Index);

   Success = (Max <= Tolerance);

   if (RelativeTolerance)
   {
      float Value = 0;

      if (Img1.Type == Img1.U8)
         Value = static_cast<const CImage<unsigned char>&>(Img1)(Index.X, Index.Y);
      else if (Img1.Type == Img1.U16)
         Value = static_cast<const CImage<unsigned short>&>(Img1)(Index.X, Index.Y);
      else if (Img1.Type == Img1.F32)
         Value = static_cast<const CImage<float>&>(Img1)(Index.X, Index.Y);
      else if (Img1.Type == Img1.F64)
         Value = static_cast<const CImage<double>&>(Img1)(Index.X, Index.Y);
      /*else
         assert(false);*/

      Success = (Max / Value) < Tolerance;
   }

   return Success;
}
