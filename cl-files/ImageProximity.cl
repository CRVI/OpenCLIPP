////////////////////////////////////////////////////////////////////////////////
//! @file	: ImageProximity.cl
//! @date   : Feb 2014
//!
//! @brief  : Pattern Matching on images
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

constant sampler_t img_prox_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define SAMPLER img_prox_sampler

#include "Images.h"

kernel void SqrDistance(INPUT source, INPUT template, OUTPUT image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float4 sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			const int2 posTemplate = { w, h };

			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;
			const int2 posSrc = { posSrcX, posSrcY };

			TYPE temp = READ_IMAGE(template, posTemplate);
			TYPE Src = READ_IMAGE(source, posSrc);
			sum += convert_float4((Src - temp)*(Src - temp));
		}
   }
		
   write_imagef(image_dest, pos, sum);
}

kernel void SqrDistance_Norm(INPUT source,INPUT template, OUTPUT image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float4 sum = 0;
   float4 sumSS = 0;
   float4 sumTT = 0;
   float4 value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			const int2 posTemplate = { w, h };

			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;
			const int2 posSrc = { posSrcX, posSrcY };

			TYPE temp = READ_IMAGE(template, posTemplate);
			TYPE Src = READ_IMAGE(source, posSrc);
			sum += CONVERT_FLOAT((Src - temp) * (Src - temp));
			sumSS += CONVERT_FLOAT(Src * Src);
			sumTT += CONVERT_FLOAT(temp * temp);
		}
   }
		
   value = sum / sqrt(sumSS * sumTT);

   write_imagef(image_dest, pos, value);
}


kernel void AbsDistance(INPUT source, INPUT template, OUTPUT image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float4 sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			const int2 posTemplate = { w, h };

			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;
			const int2 posSrc = { posSrcX, posSrcY };

			float4 temp = convert_float4(READ_IMAGE(template, posTemplate));
			float4 Src = convert_float4(READ_IMAGE(source, posSrc));
			sum += fabs(Src - temp);
		}
   }
		
   write_imagef(image_dest, pos, sum);
}

kernel void CrossCorr(INPUT source, INPUT template, OUTPUT image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float4 sum = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			const int2 posTemplate = { w, h };

			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;
			const int2 posSrc = { posSrcX, posSrcY };

			TYPE temp = READ_IMAGE(template, posTemplate);
			TYPE Src = READ_IMAGE(source, posSrc);
			sum += convert_float4(Src * temp) ;
		}
   }
		
   write_imagef(image_dest, pos, sum);
}

kernel void CrossCorr_Norm(INPUT source, INPUT template, OUTPUT image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float4 sum = 0;
   float4 sumSS = 0;
   float4 sumTT = 0;
   float4 value = 0;

   const int left = -templateWidth / 2;
   const int top  = -templateHeight / 2;

   for(int h = 0; h < templateHeight; ++h)
   {
		for(int w = 0; w < templateWidth; ++w)
		{   
			const int2 posTemplate = { w, h };

			int posSrcX = gx + w + left;
			int posSrcY = gy + h + top;
			const int2 posSrc = { posSrcX, posSrcY };

			TYPE temp = READ_IMAGE(template, posTemplate);
			TYPE Src = READ_IMAGE(source, posSrc);
			sum += CONVERT_FLOAT(Src * temp);
			sumSS += CONVERT_FLOAT(Src*Src);
			sumTT += CONVERT_FLOAT(temp*temp);
		}
   }
		
   value = sum / sqrt(sumSS * sumTT);

   write_imagef(image_dest, pos, value);
}