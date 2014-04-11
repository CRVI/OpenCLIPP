////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.cl
//! @date   : Jul 2013
//!
//! @brief  : Simple image transformation
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

#include "Images.h"

constant sampler_t lin_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
constant sampler_t rotate_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
constant sampler_t rotate_lin_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;


kernel void mirror_x(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   int width = get_image_width(source);

   WRITE_IMAGE(dest, (int2)(width - gx - 1, gy), READ_IMAGE(source, pos));
}

kernel void mirror_y(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   int height = get_image_height(source);

   WRITE_IMAGE(dest, (int2)(gx, height - gy - 1), READ_IMAGE(source, pos));
}

kernel void flip(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   int width = get_image_width(source);
   int height = get_image_height(source);

   WRITE_IMAGE(dest, (int2)(width - gx - 1, height - gy - 1), READ_IMAGE(source, pos));
}

kernel void transpose(read_only image2d_t source, write_only image2d_t dest)
{
   BEGIN

   int2 dest_pos = {pos.y, pos.x};

   WRITE_IMAGE(dest, dest_pos, READ_IMAGE(source, pos));
}

kernel void resize(read_only image2d_t source, write_only image2d_t dest, float ratioX, float ratioY)
{
   BEGIN

   float2 src_pos = {(pos.x + .4998f) * ratioX, (pos.y + .4998f) * ratioY};

   TYPE px = READ_IMAGE(source, src_pos);

   WRITE_IMAGE(dest, pos, px);
}


kernel void set_all(write_only image2d_t dest, float value)
{
   BEGIN 

   TYPE px = {value, value, value, value};
   WRITE_IMAGE(dest, pos, px);
}


#undef SAMPLER
#define SAMPLER lin_sampler

kernel void resize_linear(read_only image2d_t source, write_only image2d_t dest, float ratioX, float ratioY)
{
   BEGIN

   float2 src_pos = {(pos.x + .4995f) * ratioX, (pos.y + .4995f) * ratioY};

   TYPE px = READ_IMAGE(source, src_pos);

   WRITE_IMAGE(dest, pos, px);
}


#undef SAMPLER
#define SAMPLER rotate_sampler

kernel void rotate_img(read_only image2d_t source, write_only image2d_t dest, float sina, float cosa, float xshift, float yshift)
{
   BEGIN

   float srcx = gx - xshift;
   float srcy = gy - yshift;
   float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);

   WRITE_IMAGE(dest, pos, READ_IMAGE(source, srcpos));
}

#undef SAMPLER
#define SAMPLER rotate_lin_sampler

kernel void rotate_linear(read_only image2d_t source, write_only image2d_t dest, float sina, float cosa, float xshift, float yshift)
{
   BEGIN

   float srcx = gx - xshift;
   float srcy = gy - yshift;
   float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);

   WRITE_IMAGE(dest, pos, READ_IMAGE(source, srcpos));
}
