////////////////////////////////////////////////////////////////////////////////
//! @file	: Vector_Statistics.cl
//! @date   : Jul 2013
//!
//! @brief  : Statistical reductions on image buffers
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

#include "Buffers.h"


#define BUFFER_LENGTH 256

#define DO_REDUCE(function, index1, index2) \
   if (nb_pixels[index1] && nb_pixels[index2])\
   {\
      Weight = convert_float(nb_pixels[index2]) / nb_pixels[index1];\
      buffer[index1] = function(buffer[index1], buffer[index2], Weight);\
      nb_pixels[index1] += nb_pixels[index2];\
   }

#define WIDTH1 16  // Number of pixels per worker
#define WIDTH1_BITS 4   // Number of bits represented by WIDTH1 (8 -> 3, 16 -> 4, 32 -> 5)
#define POSI(i) (int2)(gx + i, gy)

// This version handles images of any size - it will be a bit slower
#define REDUCE(name, type, preop, fun1, postop1, fun2, postop2) \
__attribute__((reqd_work_group_size(16, 16, 1)))\
kernel void name(INPUT_SPACE const SCALAR * source, global float * result, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   local int  nb_pixels[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SCALAR);\
   float Weight;\
   \
   if (gx < img_width && gy < img_height)\
   {\
      type Res = preop(source[(gy * src_step) + gx + 0]);\
      int Nb = 1;\
      for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
         if (gx + i < img_width)\
         {\
            Res = fun1(Res, (type) preop(source[(gy * src_step) + gx + i]));\
            Nb++;\
         }\
      \
      buffer[lid] = postop1(Res, Nb);\
      nb_pixels[lid] = Nb;\
   }\
   else\
      nb_pixels[lid] = 0;\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 128)\
      DO_REDUCE(fun2, lid, lid + 128);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 64)\
      DO_REDUCE(fun2, lid, lid + 64);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 32)\
      DO_REDUCE(fun2, lid, lid + 32);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 16)\
      DO_REDUCE(fun2, lid, lid + 16);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 8)\
      DO_REDUCE(fun2, lid, lid + 8);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 4)\
      DO_REDUCE(fun2, lid, lid + 4);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 2)\
      DO_REDUCE(fun2, lid, lid + 2);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid == 0)\
   {\
      DO_REDUCE(fun2, lid, lid + 1);\
      postop2(result, buffer[0], nb_pixels[0]);\
   }\
}

// This version is for images that have a Width that is a multiple of 16*WIDTH1 and a height that is a multiple of 16
#define REDUCE_FLUSH(name, type, preop, fun1, postop1, fun2, postop2) \
__attribute__((reqd_work_group_size(16, 16, 1)))\
kernel void name(INPUT_SPACE const SCALAR * source, global float * result, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SCALAR);\
   \
   type Res = preop(source[(gy * src_step) + gx + 0]);\
   for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
      Res = fun1(Res, (type) preop(source[(gy * src_step) + gx + i]));\
   \
   buffer[lid] = postop1(Res, WIDTH1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 128)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 128], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 64)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 64], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 32)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 32], 1);\
   /* TODO : Optimize these last operations */\
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 16)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 16], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 8)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 8], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 4)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 4], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 2)\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 2], 1);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid == 0)\
   {\
      buffer[lid] = fun2(buffer[lid], buffer[lid + 1], 1);\
      postop2(result, buffer[0], 16 * 16 * WIDTH1);\
   }\
}

#define REDUCE_KERNEL(name, type, preop, fun1, postop1, fun2, postop2) \
REDUCE(name, type, preop, fun1, postop1, fun2, postop2)\
REDUCE_FLUSH(CONCATENATE(name, _flush), type, preop, fun1, postop1, fun2, postop2)

#define NOOP(a)      a
#define NOOP2(a, nb) a
#define SQR(a)       (convert_float(a) * a)
#define NO_Z(a)      (a != 0 ? 1 : 0)
#define SUM(a, b)    (a + b)
#define MEAN(a, b)   ((a + b) / 2)
#define DIV(a, b)    (a / b)

// These have a additional w parameter that is the weight of the b value, used only for MEAN2
#define MIN2(a, b, w)   min(a, b)
#define MAX2(a, b, w)   max(a, b)
#define SUM2(a, b, w)   (a + b)
#define MEAN2(a, b, w)  ((a + b * w) / (1 + w))


#define FLOAT_ATOMIC(name, fun) \
void name(global float * result, float value, int nb_pixels)\
{\
   global int * intPtr = (global int *) result;\
   bool ExchangeDone = false;\
   while (!ExchangeDone)\
   {\
      float V = *result;\
      float NewValue = fun(V, value);\
      int intResult = atomic_cmpxchg(intPtr, as_int(V), as_int(NewValue));\
      float Result = as_float(intResult);\
      ExchangeDone = (V == Result);\
   }\
}

FLOAT_ATOMIC(atomic_minf, min)
FLOAT_ATOMIC(atomic_maxf, max)


void store_value(global float * result_buffer, float value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
   const int offset = get_num_groups(0) * get_num_groups(1);
   result_buffer[gid] = value;
   result_buffer[offset + gid] = nb_pixels;
}


//            name             type    preop fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min,      SCALAR, NOOP, min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_max,      SCALAR, NOOP, max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_minabs,   SCALAR, ABS,  min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_maxabs,   SCALAR, ABS,  max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_sum,      float,  NOOP, SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_count_nz, float,  NO_Z, SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_mean,     float,  NOOP, SUM,  DIV,   MEAN2, store_value)
REDUCE_KERNEL(reduce_mean_sqr, float,  SQR,  SUM,  DIV,   MEAN2, store_value)


// Initialize result to a valid value
kernel void init(INPUT_SPACE const SCALAR * source, global float * result)
{
   *result = (float) *source;
}

kernel void init_abs(INPUT_SPACE const SCALAR * source, global float * result)
{
   *result = (float) ABS(*source);
}
