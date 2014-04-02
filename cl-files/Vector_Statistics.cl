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


// This version handles images of any size - it will be a bit slower
#define REDUCE(name, type, preop, fun1, postop1, fun2, postop2) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT_SPACE const SRC_TYPE * source, global float * result, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   local int  nb_pixels[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SRC_TYPE);\
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
#define REDUCE_FLUSH(name, type, preop, fun1, postop1, fun2, postop2) \
__attribute__((reqd_work_group_size(LW, LW, 1)))\
kernel void name(INPUT_SPACE const SRC_TYPE * source, global float * result, int src_step, int img_width, int img_height)\
{\
   local type buffer[BUFFER_LENGTH];\
   int gx1 = get_global_id(0);\
   const int gx = (gx1 & (WIDTH1 - 1)) + (gx1 >> WIDTH1_BITS) * WIDTH1 * WIDTH1;\
   const int gy = get_global_id(1);\
   const int lid = get_local_id(1) * get_local_size(0) + get_local_id(0);\
   src_step /= sizeof(SRC_TYPE);\
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

#define REDUCE_KERNEL(name, type, preop, fun1, postop1, fun2, postop2) \
REDUCE(name, type, preop, fun1, postop1, fun2, postop2)\
REDUCE_FLUSH(CONCATENATE(name, _flush), type, preop, fun1, postop1, fun2, postop2)

#define NOOP(a)      a
#define NOOP2(a, nb) a
#define SQR(a)       (a * a)
#define NO_Z(a)      (a != 0 ? 1 : 0)
#define SUM(a, b)    (a + b)
#define MEAN(a, b)   ((a + b) / 2)
#define DIV(a, b)    (a / b)

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

FLOAT_ATOMIC(atomic_minf, min)
FLOAT_ATOMIC(atomic_maxf, max)

FLOAT_ATOMIC2(atomic_minf_2C, atomic_minf)
FLOAT_ATOMIC2(atomic_maxf_2C, atomic_maxf)
FLOAT_ATOMIC3(atomic_minf_3C, atomic_minf)
FLOAT_ATOMIC3(atomic_maxf_3C, atomic_maxf)
FLOAT_ATOMIC4(atomic_minf_4C, atomic_minf)
FLOAT_ATOMIC4(atomic_maxf_4C, atomic_maxf)


// Store the partially calculated value - the final result will be calculated by the CPU
void store_value(global float * result_buffer, float value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);
   const int offset = get_num_groups(0) * get_num_groups(1);
   result_buffer[gid] = value;
   result_buffer[offset * 4 + gid] = nb_pixels;
}

void store_value_2C(global float * buffer, float2 value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

   global float2 * result_buffer = (global float2 *) (buffer + gid * 4);
   *result_buffer = value;

   const int offset = get_num_groups(0) * get_num_groups(1);
   buffer[offset * 4 + gid] = nb_pixels;
}

void store_value_3C(global float * buffer, float3 value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

   global float3 * result_buffer = (global float3 *) (buffer + gid * 4);
   *result_buffer = value;

   const int offset = get_num_groups(0) * get_num_groups(1);
   buffer[offset * 4 + gid] = nb_pixels;
}

void store_value_4C(global float * buffer, float4 value, int nb_pixels)
{
   const int gid = get_group_id(1) * get_num_groups(0) + get_group_id(0);

   global float4 * result_buffer = (global float4 *) (buffer + gid * 4);
   *result_buffer = value;

   const int offset = get_num_groups(0) * get_num_groups(1);
   buffer[offset * 4 + gid] = nb_pixels;
}

#define SRC_TYPE SCALAR

//            name             type    preop fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min,      SCALAR, NOOP, min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_max,      SCALAR, NOOP, max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_minabs,   SCALAR, ABS,  min,  NOOP2, MIN2,  atomic_minf)
REDUCE_KERNEL(reduce_maxabs,   SCALAR, ABS,  max,  NOOP2, MAX2,  atomic_maxf)
REDUCE_KERNEL(reduce_sum,      float,  NOOP, SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_sum_sqr,  float,  SQR,  SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_count_nz, float,  NO_Z, SUM,  NOOP2, SUM2,  store_value)
REDUCE_KERNEL(reduce_mean,     float,  NOOP, SUM,  DIV,   MEAN2, store_value)
REDUCE_KERNEL(reduce_mean_sqr, float,  SQR,  SUM,  DIV,   MEAN2, store_value)


#undef  SRC_TYPE
#define SRC_TYPE TYPE2

//            name                  src_type type     preop fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min_2C,        TYPE2,   NOOP, min,  NOOP2, MIN2,  atomic_minf_2C)
REDUCE_KERNEL(reduce_max_2C,        TYPE2,   NOOP, max,  NOOP2, MAX2,  atomic_maxf_2C)
REDUCE_KERNEL(reduce_minabs_2C,     TYPE2,   ABS,  min,  NOOP2, MIN2,  atomic_minf_2C)
REDUCE_KERNEL(reduce_maxabs_2C,     TYPE2,   ABS,  max,  NOOP2, MAX2,  atomic_maxf_2C)
REDUCE_KERNEL(reduce_sum_2C,        float2,  NOOP, SUM,  NOOP2, SUM2,  store_value_2C)
REDUCE_KERNEL(reduce_sum_sqr_2C,    float2,  SQR,  SUM,  NOOP2, SUM2,  store_value_2C)
REDUCE_KERNEL(reduce_mean_2C,       float2,  NOOP, SUM,  DIV,   MEAN2, store_value_2C)
REDUCE_KERNEL(reduce_mean_sqr_2C,   float2,  SQR,  SUM,  DIV,   MEAN2, store_value_2C)

#undef  SRC_TYPE
#define SRC_TYPE TYPE3

//            name                  src_type type     preop fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min_3C,        TYPE3,   NOOP, min,  NOOP2, MIN2,  atomic_minf_3C)
REDUCE_KERNEL(reduce_max_3C,        TYPE3,   NOOP, max,  NOOP2, MAX2,  atomic_maxf_3C)
REDUCE_KERNEL(reduce_minabs_3C,     TYPE3,   ABS,  min,  NOOP2, MIN2,  atomic_minf_3C)
REDUCE_KERNEL(reduce_maxabs_3C,     TYPE3,   ABS,  max,  NOOP2, MAX2,  atomic_maxf_3C)
REDUCE_KERNEL(reduce_sum_3C,        float3,  NOOP, SUM,  NOOP2, SUM2,  store_value_3C)
REDUCE_KERNEL(reduce_sum_sqr_3C,    float3,  SQR,  SUM,  NOOP2, SUM2,  store_value_3C)
REDUCE_KERNEL(reduce_mean_3C,       float3,  NOOP, SUM,  DIV,   MEAN2, store_value_3C)
REDUCE_KERNEL(reduce_mean_sqr_3C,   float3,  SQR,  SUM,  DIV,   MEAN2, store_value_3C)

#undef  SRC_TYPE
#define SRC_TYPE TYPE4

//            name                  type     preop fun1  post1  fun2  postop2
REDUCE_KERNEL(reduce_min_4C,        TYPE4,   NOOP, min,  NOOP2, MIN2,  atomic_minf_4C)
REDUCE_KERNEL(reduce_max_4C,        TYPE4,   NOOP, max,  NOOP2, MAX2,  atomic_maxf_4C)
REDUCE_KERNEL(reduce_minabs_4C,     TYPE4,   ABS,  min,  NOOP2, MIN2,  atomic_minf_4C)
REDUCE_KERNEL(reduce_maxabs_4C,     TYPE4,   ABS,  max,  NOOP2, MAX2,  atomic_maxf_4C)
REDUCE_KERNEL(reduce_sum_4C,        float4,  NOOP, SUM,  NOOP2, SUM2,  store_value_4C)
REDUCE_KERNEL(reduce_sum_sqr_4C,    float4,  SQR, SUM,  NOOP2, SUM2,  store_value_4C)
REDUCE_KERNEL(reduce_mean_4C,       float4,  NOOP, SUM,  DIV,   MEAN2, store_value_4C)
REDUCE_KERNEL(reduce_mean_sqr_4C,   float4,  SQR,  SUM,  DIV,   MEAN2, store_value_4C)


// Initialize result to a valid value
kernel void init(INPUT_SPACE const SCALAR * source, global float * result)
{
   *result = (float) *source;
}

kernel void init_abs(INPUT_SPACE const SCALAR * source, global float * result)
{
   *result = (float) ABS(*source);
}

kernel void init_2C(INPUT_SPACE const TYPE2 * source, global float2 * result)
{
   *result = convert_float2(*source);
}

kernel void init_abs_2C(INPUT_SPACE const TYPE2 * source, global float2 * result)
{
   *result = convert_float2(ABS(*source));
}

kernel void init_3C(INPUT_SPACE const TYPE3 * source, global float3 * result)
{
   *result = convert_float3(*source);
}

kernel void init_abs_3C(INPUT_SPACE const TYPE3 * source, global float3 * result)
{
   *result = convert_float3(ABS(*source));
}

kernel void init_4C(INPUT_SPACE const TYPE4 * source, global float4 * result)
{
   *result = convert_float4(*source);
}

kernel void init_abs_4C(INPUT_SPACE const TYPE4 * source, global float4 * result)
{
   *result = convert_float4(ABS(*source));
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
