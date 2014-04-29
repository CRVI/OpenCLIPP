////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.Impl.h
//! @date   : Apr 2014
//!
//! @brief  : Implementation of ImageProximity
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

// This file contains the implementation of the filters
// It will be included by Filters.cl and Vector_Filters.cl
// It requires these macros to be defined :
// SUFFIX      suffix to use for function names, like _4C
// INTERNAL    type to use for internal calculations, like float4
// INPUT       type of the input image
// OUTPUT      type of the output image
// READ        code to read 1 pixel from an image
// WRITE       code to write 1 pixel to an image
// DIM_ARGS    arguments that describe the image size (step, width & height)
// DIMS        values that describe the image size (step, width & height)


kernel void CONCATENATE(SqrDistance, SUFFIX)(INPUT source, INPUT templ, OUTPUT dest, DIM_ARGS)
{
   BEGIN

   private INTERNAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;

			INTERNAL temp = CONVERT_INTERNAL(READ(templ, temp_step, w, h));
         INTERNAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = CONVERT_INTERNAL(READ(source, src_step, posSrcX, posSrcY));
         }
			sum += (Src - temp)*(Src - temp);
		}
   }

   WRITE(dest, sum);
}

kernel void CONCATENATE(SqrDistance_Norm, SUFFIX)(INPUT source, INPUT templ, OUTPUT dest, DIM_ARGS)
{
   BEGIN

   INTERNAL sum = 0;
   INTERNAL sumSS = 0;
   INTERNAL sumTT = 0;
   INTERNAL value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;

         INTERNAL temp = CONVERT_INTERNAL(READ(templ, temp_step, w, h));
         INTERNAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = CONVERT_INTERNAL(READ(source, src_step, posSrcX, posSrcY));
         }
			sum += (Src - temp) * (Src - temp);
			sumSS += Src * Src;
			sumTT += temp * temp;
		}
   }

   value = sum / sqrt(sumSS * sumTT);

   WRITE(dest, value);
}

kernel void CONCATENATE(AbsDistance, SUFFIX)(INPUT source, INPUT templ, OUTPUT dest, DIM_ARGS)
{
   BEGIN

   INTERNAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;

			INTERNAL temp = CONVERT_INTERNAL(READ(templ, temp_step, w, h));
         INTERNAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = CONVERT_INTERNAL(READ(source, src_step, posSrcX, posSrcY));
         }
			sum += fabs(Src - temp);
		}
   }

   WRITE(dest, sum);
}

kernel void CONCATENATE(CrossCorr, SUFFIX)(INPUT source, INPUT templ, OUTPUT dest, DIM_ARGS)
{
   BEGIN

   INTERNAL sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;

         INTERNAL temp = CONVERT_INTERNAL(READ(templ, temp_step, w, h));
         INTERNAL Src = 0;

         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = CONVERT_INTERNAL(READ(source, src_step, posSrcX, posSrcY));
         }
			sum += Src * temp ;
		}
   }

   WRITE(dest, sum);
}

kernel void CONCATENATE(CrossCorr_Norm, SUFFIX)(INPUT source, INPUT templ, OUTPUT dest, DIM_ARGS)
{
   BEGIN

   INTERNAL sum = 0;
   INTERNAL sumSS = 0;
   INTERNAL sumTT = 0;
   INTERNAL value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;

         INTERNAL temp = CONVERT_INTERNAL(READ(templ, temp_step, w, h));
         INTERNAL Src = 0;


         if(posSrcX < 0 || posSrcY < 0 || posSrcX >= dest_width || posSrcY >= dest_height)
         {
            Src = 0;
         }
         else
         {
            Src = CONVERT_INTERNAL(READ(source, src_step, posSrcX, posSrcY));
         }
			sum += Src * temp ;
         sumSS += Src * Src;
			sumTT += temp * temp;
		}
   }

   value = sum / sqrt(sumSS * sumTT);

   WRITE(dest, value);
}