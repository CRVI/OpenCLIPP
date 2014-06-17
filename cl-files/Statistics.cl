////////////////////////////////////////////////////////////////////////////////
//! @file	: Statistics.cl
//! @date   : Jul 2013
//!
//! @brief  : Statistical reductions
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


#define DO_REDUCE(function, index1, index2) \
   if (nb_pixels[index1] && nb_pixels[index2])\
   {\
      Weight = convert_float(nb_pixels[index2]) / nb_pixels[index1];\
      buffer[index1] = function(buffer[index1], buffer[index2], Weight);\
      nb_pixels[index1] += nb_pixels[index2];\
   }

#define LW 16                    // Local width - workgroups are 2D squares of LW*LW
#define BUFFER_LENGTH (LW * LW)  // Size of a local buffer with 1 item per workitem

#define WIDTH1 16       // Number of pixels per workitem
#define WIDTH1_BITS 4   // Number of bits represented by WIDTH1 (8 -> 3, 16 -> 4, 32 -> 5)

#define POSI(i) (int2)(gx + i, gy)

#define INPUT INPUT_SPACE const TYPE *

// This version handles images of any size - it will be a bit slower
#define REDUCE(name, type, preop, fun1, postop1, fun2, postop2, param) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT source, global float * result, int src_step, int img_width, int img_height param() )\
{\
   local type buffer[BUFFER_LENGTH];\
   local int  nb_pixels[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(TYPE);\
   float Weight;\
   \
   if (gx < img_width && gy < img_height)\
   {\
      type Res = CONCATENATE(convert_, type)(source[(gy * src_step) + gx + 0]);\
      Res = preop(Res);\
      int Nb = 1;\
      for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
         if (gx + i < img_width)\
         {\
            type px = CONCATENATE(convert_, type)(source[(gy * src_step) + gx + i]);\
            Res = fun1(Res, preop(px));\
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

// This version is for images that have a Width that is a multiple of LW*WIDTH1 and a height that is a multiple of LW
#define REDUCE_FLUSH(name, type, preop, fun1, postop1, fun2, postop2, param) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT source, global float * result, int src_step, int img_width, int img_height param() )\
{\
   local type buffer[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(TYPE);\
   \
   type Res = CONCATENATE(convert_, type)(source[(gy * src_step) + gx + 0]);\
   Res = preop(Res);\
   for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
   {\
      type px = CONCATENATE(convert_, type)(source[(gy * src_step) + gx + i]);\
      Res = fun1(Res, preop(px));\
   }\
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
      postop2(result, buffer[0], LW * LW * WIDTH1);\
   }\
}

#define EMPTY()
#define AVG_PARAM() , float4 Avg

#define REDUCE_KERNEL(name, type, preop, fun1, postop1, fun2, postop2) \
REDUCE(name, type, preop, fun1, postop1, fun2, postop2, EMPTY)\
REDUCE_FLUSH(CONCATENATE(name, _flush), type, preop, fun1, postop1, fun2, postop2, EMPTY)

#define REDUCE_KERNEL_P(name, type, preop, fun1, postop1, fun2, postop2, param) \
REDUCE(name, type, preop, fun1, postop1, fun2, postop2, param)\
REDUCE_FLUSH(CONCATENATE(name, _flush), type, preop, fun1, postop1, fun2, postop2, param)

#define NOOP(a)      a
#define NOOP2(a, nb) a
#define SQR(a)       (a * a)
#define NO_Z(a)      CONVERT_REAL(a != 0 ? 1 : 0)
#define SUM(a, b)    (a + b)
#define MEAN(a, b)   ((a + b) / 2)
#define DIV(a, b)    (a / b)
#define SQAVGD(a)    (a - SEL_CHAN(Avg)) * (a - SEL_CHAN(Avg))

// These have a additional w parameter that is the weight of the b value, used only for MEAN2
#define MIN2(a, b, w)   min(a, b)
#define MAX2(a, b, w)   max(a, b)
#define SUM2(a, b, w)   (a + b)
#define MEAN2(a, b, w)  ((a + b * w) / (1 + w))


// OpenCL does not provide atomic float operations
// This simulates an atomic operation on a float by using atomic_cmpxchg()
// The loop in there is kinda bad and would loop many times if the concurency is high
// But because this code is called when only a small number of work-items are running (max of 1 per workgroup)
// it works well.
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

#define FLOAT_ATOMIC2(name, name1C) \
void name(global float * result, TYPE2 value, int nb_pixels)\
{\
   name1C(result + 0, value.x, nb_pixels);\
   name1C(result + 1, value.y, nb_pixels);\
}

#define FLOAT_ATOMIC3(name, name1C) \
void name(global float * result, TYPE3 value, int nb_pixels)\
{\
   name1C(result + 0, value.x, nb_pixels);\
   name1C(result + 1, value.y, nb_pixels);\
   name1C(result + 2, value.z, nb_pixels);\
}

#define FLOAT_ATOMIC4(name, name1C) \
void name(global float * result, TYPE4 value, int nb_pixels)\
{\
   name1C(result + 0, value.x, nb_pixels);\
   name1C(result + 1, value.y, nb_pixels);\
   name1C(result + 2, value.z, nb_pixels);\
   name1C(result + 3, value.w, nb_pixels);\
}

FLOAT_ATOMIC(atomic_minf_1, min)
FLOAT_ATOMIC(atomic_maxf_1, max)

FLOAT_ATOMIC2(atomic_minf_2, atomic_minf_1)
FLOAT_ATOMIC2(atomic_maxf_2, atomic_maxf_1)
FLOAT_ATOMIC3(atomic_minf_3, atomic_minf_1)
FLOAT_ATOMIC3(atomic_maxf_3, atomic_maxf_1)
FLOAT_ATOMIC4(atomic_minf_4, atomic_minf_1)
FLOAT_ATOMIC4(atomic_maxf_4, atomic_maxf_1)

#ifdef NBCHAN
#define atomic_minf CONCATENATE(atomic_minf_, NBCHAN)
#define atomic_maxf CONCATENATE(atomic_maxf_, NBCHAN)
#else
#define atomic_minf atomic_minf_1
#define atomic_maxf atomic_maxf_1
#endif

// Store the partially calculated value - the final result will be calculated by the CPU
void store_value(global float * buffer, REAL value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

   global REAL * result_buffer = (global REAL *) (buffer + gid * 4);
   *result_buffer = value;

   const int offset = get_num_groups(0) * get_num_groups(1);
   buffer[offset * 4 + gid] = nb_pixels;
}


#ifndef NBCHAN
#define SEL_CHAN(code) code.x
#else
#if NBCHAN == 2
#define SEL_CHAN(code) code.xy
#else
#if NBCHAN == 3
#define SEL_CHAN(code) code.xyz
#else
#define SEL_CHAN(code) code.xyzw
#endif
#endif
#endif

//            name             type    preop    fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min,      TYPE,   NOOP,    min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_max,      TYPE,   NOOP,    max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_minabs,   TYPE,   ABS,     min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_maxabs,   TYPE,   ABS,     max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_sum,      REAL,   NOOP,    SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_sum_sqr,  REAL,   SQR,     SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_count_nz, REAL,   NO_Z,    SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_mean,     REAL,   NOOP,    SUM,  DIV,   MEAN2, store_value)
REDUCE_KERNEL(reduce_mean_sqr, REAL,   SQR,     SUM,  DIV,   MEAN2, store_value)
REDUCE_KERNEL_P(reduce_stddev, REAL,   SQAVGD,  SUM,  DIV,   MEAN2, store_value, AVG_PARAM)


// Initialize result to a valid value
kernel void init(INPUT source, global REAL * result)
{
   *result = CONVERT_REAL(*source);
}

kernel void init_abs(INPUT source, global REAL * result)
{
   *result = CONVERT_REAL(ABS(*source));
}


// DO_REDUCE that handles coordinates
#undef DO_REDUCE
#define DO_REDUCE(compare, index1, index2) \
   if (nb_pixels[index1] && nb_pixels[index2])\
   {\
      if (buffer[index2] compare buffer[index1])\
      {\
         buffer[index1] = buffer[index2];\
         coord_buf[index1] = coord_buf[index2];\
      }\
   }

// This version finds the X and Y coordinates of the min or max value
// Only for 1 channel images
#define REDUCE_POS_STD(name, type, preop, comp) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT_SPACE const SCALAR * source, global float * result, global int2 * result_coord, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   local int2 coord_buf[BUFFER_LENGTH];\
   local char nb_pixels[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SCALAR);\
   \
   if (gx < img_width && gy < img_height)\
   {\
      type Res = preop(source[(gy * src_step) + gx + 0]);\
      int2 coord = (int2)(gx, gy);\
      int Nb = 1;\
      for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
         if (gx + i < img_width)\
         {\
            type px = preop(source[(gy * src_step) + gx + i]);\
            if (px comp Res)\
            {\
               Res = px;\
               coord.x = gx + i;\
            }\
            \
         }\
      \
      buffer[lid] = Res;\
      coord_buf[lid] = coord;\
      nb_pixels[lid] = 1;\
   }\
   else\
      nb_pixels[lid] = 0;\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 128)\
      DO_REDUCE(comp, lid, lid + 128);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 64)\
      DO_REDUCE(comp, lid, lid + 64);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 32)\
      DO_REDUCE(comp, lid, lid + 32);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 16)\
      DO_REDUCE(comp, lid, lid + 16);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 8)\
      DO_REDUCE(comp, lid, lid + 8);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 4)\
      DO_REDUCE(comp, lid, lid + 4);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 2)\
      DO_REDUCE(comp, lid, lid + 2);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid == 0)\
   {\
      DO_REDUCE(comp, 0, 1);\
      const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);\
      result[gid] = buffer[0];\
      result_coord[gid] = coord_buf[0];\
   }\
}

#define DO_REDUCE_FLUSH(compare, index1, index2) \
   if (buffer[index2] compare buffer[index1])\
   {\
      buffer[index1] = buffer[index2];\
      coord_buf[index1] = coord_buf[index2];\
   }\


// This version finds the X and Y coordinates in a faster way for flush images
#define REDUCE_POS_FLUSH(name, type, preop, comp) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT_SPACE const SCALAR * source, global float * result, global int2 * result_coord, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   local int2 coord_buf[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SCALAR);\
   type Res = preop(source[(gy * src_step) + gx + 0]);\
   int2 coord = (int2)(gx, gy);\
   for (int i = WIDTH1; i < WIDTH1 * WIDTH1; i += WIDTH1)\
   {\
      type px = preop(source[(gy * src_step) + gx + i]);\
      if (px comp Res)\
      {\
         Res = px;\
         coord.x = gx + i;\
      }\
      \
   }\
   \
   buffer[lid] = Res;\
   coord_buf[lid] = coord;\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 128)\
      DO_REDUCE_FLUSH(comp, lid, lid + 128);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 64)\
      DO_REDUCE_FLUSH(comp, lid, lid + 64);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 32)\
      DO_REDUCE_FLUSH(comp, lid, lid + 32);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 16)\
      DO_REDUCE_FLUSH(comp, lid, lid + 16);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 8)\
      DO_REDUCE_FLUSH(comp, lid, lid + 8);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 4)\
      DO_REDUCE_FLUSH(comp, lid, lid + 4);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid < 2)\
      DO_REDUCE_FLUSH(comp, lid, lid + 2);\
   \
   barrier(CLK_LOCAL_MEM_FENCE);\
   \
   if (lid == 0)\
   {\
      DO_REDUCE_FLUSH(comp, 0, 1);\
      const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);\
      result[gid] = buffer[0];\
      result_coord[gid] = coord_buf[0];\
   }\
}

#define REDUCE_POS(name, type, preop, comp) \
   REDUCE_POS_FLUSH(CONCATENATE(name, _flush), type, preop, comp)\
   REDUCE_POS_STD(name, type, preop, comp)

//         name           type    preop comp
REDUCE_POS(min_coord,     SCALAR, NOOP, <)
REDUCE_POS(max_coord,     SCALAR, NOOP, >)
REDUCE_POS(min_abs_coord, SCALAR, ABS,  <)
REDUCE_POS(max_abs_coord, SCALAR, ABS,  >)
