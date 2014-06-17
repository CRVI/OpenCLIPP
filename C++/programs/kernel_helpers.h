////////////////////////////////////////////////////////////////////////////////
//! @file	: kernel_helpers.h
//! @date   : Jul 2013
//!
//! @brief  : Macros to simplify kernel calls
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

/* This file provides macros for calling OpenCL kernels 
   It is intended to be used by descendants of Program

   Calling a kernel is normally done like this using cl.hpp :
   auto functor = cl::make_kernel<cl::Image2D, cl::Image2D, cl::Image2D, float, int>(program, "kernel_name");
   functor(cl::EnqueueArgs(command_queue, src_img1.FullRange()), src_img1, src_img2, dst_img, argument1, argument2);

   Using the macro, it is done like this :
   Kernel(kernel_name, In(src_img1, src_img2), Out(dst_img), argument1, argument2);

   The macro also calls :
      SendIfNeeded() on the sources to send them automatically to the device
      SetInDevice() on the destinations to mark them as containing useful data

   The kernel is enqueued in the the command queue of the OpenCL object.
   This means execution is not done immediately, it will be done when the GPU is available and in parallel with execution of the rest of the host program.
   The macro does not transfer the outputs to the host, they stay in the device.
   To wait for the execution of the kernel and transfer the outputs back to the host,
      call Read(true) on the outputs after calling the macro : dst_img.Read(true);

   At least 1 source is needed. The first source needs to be an image (derived from ImageBase)
   The kernels need to have their arguments with the sources first, destinations second and other args afterward
   The Kernel() macro needs to be used with the same argument order and types as the kernels in the .cl files
   In() and Out() can be omitted if only 1 source/destination is used.

   Usage examples :
    with 1 Source, 1 Dest :
      Kernel(name, Source, Dest);

    with 1 Source, 1 Dest, 2 Args :
      Kernel(name, Source, Dest, arg1, arg2);

    with 2 Sources, 1 Dest :
      Kernel(name, In(Source1, Source2), Dest);

    with 1 Source, 0 dest, 2 Args :
      Kernel(name, Source, Out(), arg1, arg2);
*/

#include "../preprocessor.h"

namespace OpenCLIPP
{

// SelectClType<T>, SET_CL_TYPE() and CL_TYPE() are used to find the proper type to use with cl::make_kernel
template<class T>
struct SelectClType
{
   typedef T Type;
};

#define SET_CL_TYPE(type, cl_type)\
   template<> struct SelectClType<type>\
   {   typedef cl_type Type; };\
   template<> struct SelectClType<const type>\
   {   typedef cl_type Type; };\
   template<> struct SelectClType<type&>\
   {   typedef cl_type Type; };\
   template<> struct SelectClType<const type&>\
   {   typedef cl_type Type; };

SET_CL_TYPE(IBuffer,     cl::Buffer)
SET_CL_TYPE(Buffer,      cl::Buffer)
SET_CL_TYPE(ReadBuffer,  cl::Buffer)
SET_CL_TYPE(TempBuffer,  cl::Buffer)
SET_CL_TYPE(Image,       cl::Buffer)
SET_CL_TYPE(TempImage,   cl::Buffer)


#define CL_TYPE(arg) SelectClType<decltype(arg)>::Type


// These macros are customizable to allow usage of the macro to call kernels that work differently
#ifndef SELECT_PROGRAM
#define SELECT_PROGRAM(...) SelectProgram(__VA_ARGS__)
#endif   // SELECT_PROGRAM

#ifndef SELECT_NAME
#define SELECT_NAME(name, src_img) #name
#endif   // SELECT_NAME

#ifndef KERNEL_RANGE
#define KERNEL_RANGE(...) _FIRST(__VA_ARGS__).FullRange()
#endif   // SELECT_NAME

#ifndef LOCAL_RANGE
#define LOCAL_RANGE     // Defaults to no local rage
#endif   // LOCAL_RANGE


// In and Out macros are used to mark the input and output of the kernels
// At least 1 input is needed, output may be empty : Out()
// The first input must be derived from ImageBase
#define In(...) __VA_ARGS__   ///< Used to mark inputs to a kernel
#define Out(...) __VA_ARGS__  ///< Used to mark outputs to a kernel


/// Simple kernel calling macro.
/// Usage : Kernel(name, In(Src1, Src2), Out(Dst), IntArg, FloatAgr);
#define Kernel(name, in, out, ...) Kernel_(*m_CL, SELECT_PROGRAM(in ADD_COMMA(out) out), name, KERNEL_RANGE(in ADD_COMMA(out) out), LOCAL_RANGE, In(in), Out(out), __VA_ARGS__)


// Helpers
#define _SEND_IF_NEEDED(img) (img).SendIfNeeded();
#define _SET_IN_DEVICE(img) (img).SetInDevice();
#define _FIRST(in, ...) REMOVE_PAREN(SELECT_FIRST, (in))

/// More generic kernel calling macro.
/// Example usage : Kernel_(CL, ArithmeticProgram, "Add", Src1.FullRange(), cl::NDRange(16, 16, 1), In(Src1, Src2), Out(Dst), Arg1, Arg2);
#define Kernel_(CL, program, name, global_range, local_range, in, out, ...)\
   FOR_EACH(_SEND_IF_NEEDED, in)\
   cl::make_kernel<FOR_EACH_COMMA(CL_TYPE, in) ADD_COMMA(out) FOR_EACH_COMMA(CL_TYPE, out) ADD_COMMA(__VA_ARGS__) FOR_EACH_COMMA(CL_TYPE, __VA_ARGS__)>\
      ((cl::Program) program, SELECT_NAME(name, _FIRST(in)))\
         (cl::EnqueueArgs(CL, global_range ADD_COMMA(local_range) local_range),\
            in ADD_COMMA(out) out ADD_COMMA(__VA_ARGS__) __VA_ARGS__);\
   FOR_EACH(_SET_IN_DEVICE, out)

}
