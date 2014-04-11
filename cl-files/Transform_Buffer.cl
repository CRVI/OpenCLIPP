////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform_Buffer.cl
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformation on image buffers
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

#define BEGIN \
   const int gx = get_global_id(0);\
   const int gy = get_global_id(1);\
   const int2 pos = { gx, gy };

#define _UNARY_OP(name, code, type, suffix) \
   kernel void CONCATENATE(name, suffix) (INPUT_SPACE const type * source, global type * dest, int src_step, int dst_step, int width, int height)\
   {\
      BEGIN\
      src_step /= sizeof(type);\
      dst_step /= sizeof(type);\
      code;\
   }

#define UNARY_OP(name, code) \
   _UNARY_OP(name, code, SCALAR, _1C)\
   _UNARY_OP(name, code, TYPE2,  _2C)\
   _UNARY_OP(name, code, TYPE3,  _3C)\
   _UNARY_OP(name, code, TYPE4,  _4C)\

UNARY_OP(mirror_x, dest[gy * dst_step + gx] = source[gy * src_step + width - gx - 1])
UNARY_OP(mirror_y, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + gx])
UNARY_OP(flip, dest[gy * dst_step + gx] = source[(height - gy - 1) * src_step + width - gx - 1])
UNARY_OP(transpose, dest[gx * dst_step + gy] = source[gy * src_step + gx])

kernel void set_all_1C(global SCALAR * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(SCALAR);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_2C(global TYPE2 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE2);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_3C(global TYPE3 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE3);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}

kernel void set_all_4C(global TYPE4 * dest, int dst_step, float value)
{
   BEGIN
   dst_step /= sizeof(TYPE4);
   dest[gy * dst_step + gx] = CONVERT_SCALAR(value);
}


/*kernel void resize(INPUT source, OUTPUT dest, float ratioX, float ratioY)
{
   BEGIN

   float2 src_pos = {(pos.x + .4998f) * ratioX, (pos.y + .4998f) * ratioY};

   TYPE px = READ_IMAGE(source, src_pos);

   WRITE_IMAGE(dest, pos, px);
}


kernel void set_all(OUTPUT dest, float value)
{
   BEGIN 

   TYPE px = {value, value, value, value};
   WRITE_IMAGE(dest, pos, px);
}


#undef SAMPLER
#define SAMPLER lin_sampler

kernel void resize_linear(INPUT source, OUTPUT dest, float ratioX, float ratioY)
{
   BEGIN

   float2 src_pos = {(pos.x + .4995f) * ratioX, (pos.y + .4995f) * ratioY};

   TYPE px = READ_IMAGE(source, src_pos);

   WRITE_IMAGE(dest, pos, px);
}


#undef SAMPLER
#define SAMPLER rotate_sampler

kernel void rotate_img(INPUT source, OUTPUT dest, float sina, float cosa, float xshift, float yshift)
{
   BEGIN

   float srcx = gx - xshift;
   float srcy = gy - yshift;
   float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);

   WRITE_IMAGE(dest, pos, READ_IMAGE(source, srcpos));
}

#undef SAMPLER
#define SAMPLER rotate_lin_sampler

kernel void rotate_linear(INPUT source, OUTPUT dest, float sina, float cosa, float xshift, float yshift)
{
   BEGIN

   float srcx = gx - xshift;
   float srcy = gy - yshift;
   float2 srcpos = (float2)(cosa * srcx - sina * srcy + .5f, sina * srcx + cosa * srcy + .5f);

   WRITE_IMAGE(dest, pos, READ_IMAGE(source, srcpos));
}*/
