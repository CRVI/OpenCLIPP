////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_ImageProximity.cl
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on image buffers
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

#include "Buffers.h"

#define INPUT     global const TYPE *
#define OUTPUT    global REAL *

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   src_step  /= sizeof(TYPE);\
   temp_step /= sizeof(TYPE);\
   dst_step  /= sizeof(REAL);

#define READ(img, step, x, y)  CONVERT_REAL(img[y * step + x])
#define WRITE(img, val) img[gy * dst_step + gx] = val


kernel void SqrDistance(INPUT source, INPUT templ, OUTPUT dest,
                        int src_step, int temp_step, int dst_step,
                        int templateWidth, int templateHeight, int dest_width, int dest_height)
{
   BEGIN

   REAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for (int h = 0; h < templateHeight; ++h)
      for (int w = 0; w < templateWidth; ++w)
      {   
         int posSrcX = gx + w + left;
         int posSrcY = gy + h + top;

         REAL temp = READ(templ, temp_step, w, h);
         REAL Src = 0;

         if (posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
            Src = 0;
         else
            Src = READ(source, src_step, posSrcX, posSrcY);

         sum += (Src - temp) * (Src - temp);
      }

   WRITE(dest, sum);
}

kernel void SqrDistance_Norm(INPUT source, INPUT templ, OUTPUT dest,
                             int src_step, int temp_step, int dst_step,
                             int templateWidth, int templateHeight, int dest_width, int dest_height)
{
   BEGIN

   REAL sum = 0;
   REAL sumSS = 0;
   REAL sumTT = 0;
   REAL value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for (int h = 0; h < templateHeight; ++h)
      for (int w = 0; w < templateWidth; ++w)
      {   
         int posSrcX = gx + w + left;
         int posSrcY = gy + h + top;

         REAL temp = READ(templ, temp_step, w, h);
         REAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
            Src = 0;
         else
            Src = READ(source, src_step, posSrcX, posSrcY);

         sum += (Src - temp) * (Src - temp);
         sumSS += Src * Src;
         sumTT += temp * temp;
      }

   value = sum / sqrt(sumSS * sumTT);

   WRITE(dest, value);
}

kernel void AbsDistance(INPUT source, INPUT templ, OUTPUT dest,
                        int src_step, int temp_step, int dst_step,
                        int templateWidth, int templateHeight, int dest_width, int dest_height)
{
   BEGIN

   REAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
      for(int w = 0; w < templateWidth; ++w)
      {   
         int posSrcX = gx + w + left;
         int posSrcY = gy + h + top;

         REAL temp = READ(templ, temp_step, w, h);
         REAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
            Src = 0;
         else
            Src = READ(source, src_step, posSrcX, posSrcY);

         sum += fabs(Src - temp);
      }

   WRITE(dest, sum);
}

kernel void CrossCorr(INPUT source, INPUT templ, OUTPUT dest,
                      int src_step, int temp_step, int dst_step,
                      int templateWidth, int templateHeight, int dest_width, int dest_height)
{
   BEGIN

   REAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
      for(int w = 0; w < templateWidth; ++w)
      {   
         int posSrcX = gx + w + left;
         int posSrcY = gy + h + top;

         REAL temp = READ(templ, temp_step, w, h);
         REAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = READ(source, src_step, posSrcX, posSrcY);
         }

         sum += Src * temp ;
      }

   WRITE(dest, sum);
}

kernel void CrossCorr_Norm(INPUT source, INPUT templ, OUTPUT dest,
                           int src_step, int temp_step, int dst_step,
                           int templateWidth, int templateHeight, int dest_width, int dest_height)
{
   BEGIN

   REAL sum = 0;
   REAL sumSS = 0;
   REAL sumTT = 0;
   REAL value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
      for(int w = 0; w < templateWidth; ++w)
      {   
         int posSrcX = gx + w + left;
         int posSrcY = gy + h + top;

         REAL temp = READ(templ, temp_step, w, h);
         REAL Src = 0;


         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
            Src = 0;
         else
            Src = READ(source, src_step, posSrcX, posSrcY);

         sum += Src * temp ;
         sumSS += Src * Src;
         sumTT += temp * temp;
      }

   value = sum / sqrt(sumSS * sumTT);

   WRITE(dest, value);
}
