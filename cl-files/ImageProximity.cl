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

kernel void SqrDistance(read_only image2d_t source, read_only image2d_t template, write_only image2d_t image_dest, int src_width, int src_height, int templateWidth, int templateHeight)
{
   BEGIN

   float sum = 0;
   float sumSS = 0;
   float sumTT = 0;
   float value = 0;

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

			SCALAR temp = READ_IMAGE(template, posTemplate).x;
			SCALAR Src = READ_IMAGE(source, posSrc).x;
			sum += (float)((Src - temp)*(Src - temp));
			sumSS += (float)(Src*Src);
			sumTT += (float)(temp*temp);
		}
   }
		
   if(sumSS != 0 && sumTT != 0)
		value = sum / sqrt(sumSS * sumTT);

   write_imagef(image_dest, pos, value);
}
