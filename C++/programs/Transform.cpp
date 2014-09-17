////////////////////////////////////////////////////////////////////////////////
//! @file	: Transform.cpp
//! @date   : Apr 2014
//!
//! @brief  : Simple image transformations
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

#include "Programs/Transform.h"


#include "kernel_helpers.h"

// Transpose uses a local array and a local size of 32*8
#define PIXELS_PER_WORKITEM_V   4
#define LOCAL_WIDTH  32
#define LOCAL_HEIGHT 8

#include "WorkGroup.h"

#include <cmath>

using namespace std;


static const double Pi = atan(1) * 4;


namespace OpenCLIPP
{

void Transform::MirrorX(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_x, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Transform::MirrorY(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_y, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Transform::Flip(Image& Source, Image& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(flip, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void Transform::Transpose(Image& Source, Image& Dest)
{
   if (Dest.Width() < Source.Height() || Dest.Height() < Source.Width())
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "Destination image too small to receive Transform::Transpose");

   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   if (IsFlushImage(Source))
   {
      Kernel_Local(transpose_flush, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
   }
   else
   {
      Kernel_Local(transpose, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
   }

}

void Transform::Rotate(Image& Source, Image& Dest,
      double Angle, double XShift, double YShift, EInterpolationType Interpolation)
{
   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   // Convert to radians
   Angle *= Pi / 180.;

   float cosa = (float) cos(Angle);
   float sina = (float) sin(Angle);
   float xshift = (float) XShift;
   float yshift = (float) YShift;

   const SImage& SrcImg = Source;
   const SImage& DstImg = Dest;

   switch (Interpolation)
   {
   case NearestNeighbour:
      Kernel_(*m_CL, SelectProgram(Source), rotate_nn, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         sina, cosa, xshift, yshift);
      break;
   case Linear:
      Kernel_(*m_CL, SelectProgram(Source), rotate_linear, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         sina, cosa, xshift, yshift);
      break;
   case BestQuality:
   case Cubic:
      Kernel_(*m_CL, SelectProgram(Source), rotate_bicubic, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         sina, cosa, xshift, yshift);
      break;
   case Lanczos2:
   case Lanczos3:
   case SuperSampling:
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Rotate");
   }

}


int find_lanczos_buffer_size(int length)
{
   int size = 512;
   while (size < length)
      size *= 2;

   return size;
}

void Transform::ResizeLanczos(Image& Source, Image& Dest, int a, cl::NDRange Range)
{
   float RatioX = Source.Width() * 1.f / Dest.Width();
   float RatioY = Source.Height() * 1.f / Dest.Height();

   const SImage& SrcImg = Source;
   const SImage& DstImg = Dest;

   uint length = max(DstImg.Width, DstImg.Height);
   int size = find_lanczos_buffer_size(length);

   TempBuffer factors(*m_CL, size * 2 * a * 2 * sizeof(float));   // 2 lists of factors containing 2*a*size items

   cl::make_kernel<cl::Buffer, float, float, int>
      ((cl::Program) SelectProgram(Source), "prepare_resize_lanczos" + to_string(a))
         (cl::EnqueueArgs(*m_CL, cl::NDRange(length, a * 2, 2)),
            factors, RatioX, RatioY, size);

   switch (a)
   {
   case 2:
      Kernel_(*m_CL, SelectProgram(Source), resize_lanczos2, Range, LOCAL_RANGE,
         In(Source, factors), Out(Dest), SrcImg, DstImg, RatioX, RatioY, size);
      break;
   case 3:
      Kernel_(*m_CL, SelectProgram(Source), resize_lanczos3, Range, LOCAL_RANGE,
         In(Source, factors), Out(Dest), SrcImg, DstImg, RatioX, RatioY, size);
      break;
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "unsupported lanczos size");
   }

}

void Transform::Resize(Image& Source, Image& Dest, EInterpolationType Interpolation, bool KeepRatio)
{
   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   float RatioX = Source.Width() * 1.f / Dest.Width();
   float RatioY = Source.Height() * 1.f / Dest.Height();

   cl::NDRange Range = Dest.FullRange();

   if (KeepRatio)
   {
      float Ratio = max(RatioX, RatioY);
      RatioX = Ratio;
      RatioY = Ratio;

      Range = cl::NDRange(size_t(Source.Width() / RatioX), size_t(Source.Height() / RatioY), 1);
   }

   const SImage& SrcImg = Source;
   const SImage& DstImg = Dest;

   switch (Interpolation)
   {
   case NearestNeighbour:
      Kernel_(*m_CL, SelectProgram(Source), resize_nn, Range, LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case Linear:
      Kernel_(*m_CL, SelectProgram(Source), resize_linear, Range, LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case Cubic:
      Kernel_(*m_CL, SelectProgram(Source), resize_bicubic, Range, LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case SuperSampling:
      if (RatioX < 1 || RatioY < 1)
         throw cl::Error(CL_INVALID_ARG_VALUE, "Supersampling can only be used to downsize an image");

      Kernel_(*m_CL, SelectProgram(Source), resize_supersample, Range, LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case Lanczos2:
      ResizeLanczos(Source, Dest, 2, Range);
      break;
   case Lanczos3:
      ResizeLanczos(Source, Dest, 3, Range);
      break;
   case BestQuality:
      if (RatioX < 1 || RatioY < 1)
      {
         // Enlarging image, bicubic is best
         Resize(Source, Dest, Cubic, KeepRatio);
         break;
      }

      // Shrinking image, SuperSampling is best
      Resize(Source, Dest, SuperSampling, KeepRatio);
      break;
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Resize");
   }

}

void Transform::Shear(Image& Source, Image& Dest, double ShearX, double ShearY,
                      double XShift, double YShift, EInterpolationType Interpolation)
{
   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   float shearx = (float) ShearX;
   float sheary = (float) ShearY;
   float xshift = (float) XShift;
   float yshift = (float) YShift;
   float factor = 1 / (1 - shearx * sheary);

   const SImage& SrcImg = Source;
   const SImage& DstImg = Dest;

   switch (Interpolation)
   {
   case NearestNeighbour:
      Kernel_(*m_CL, SelectProgram(Source), shear_nn, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         shearx, sheary, factor, xshift, yshift);
      break;
   case Linear:
      Kernel_(*m_CL, SelectProgram(Source), shear_linear, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         shearx, sheary, factor, xshift, yshift);
      break;
   case BestQuality:
   case Cubic:
      Kernel_(*m_CL, SelectProgram(Source), shear_cubic, Dest.FullRange(), LOCAL_RANGE,
         Source, Dest, SrcImg, DstImg,
         shearx, sheary, factor, xshift, yshift);
      break;
   case Lanczos2:
   case Lanczos3:
   case SuperSampling:
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Rotate");
   }

}

void Transform::Remap(Image& Source, Image& MapX, Image& MapY, Image& Dest, EInterpolationType Interpolation)
{
   if (!SameType(Source, Dest))
      throw cl::Error(CL_INVALID_VALUE, "Different image types used");

   if (MapX.NbChannels() > 1 || MapY.NbChannels() > 1)
      throw cl::Error(CL_INVALID_VALUE, "Remap can use only images with 1 channel for the map images");

   if (MapX.DataType() != SImage::F32 || MapY.DataType() != SImage::F32)
      throw cl::Error(CL_INVALID_VALUE, "Remap can use only F32 images for the map images");

   const SImage& SrcImg = Source;
   const SImage& DstImg = Dest;

   switch (Interpolation)
   {
   case NearestNeighbour:
      Kernel_(*m_CL, SelectProgram(Source), remap_nn, Dest.FullRange(), LOCAL_RANGE,
         In(Source, MapX, MapY), Dest, MapX.Step(), MapY.Step(), SrcImg, DstImg);
      break;
   case Linear:
      Kernel_(*m_CL, SelectProgram(Source), remap_linear, Dest.FullRange(), LOCAL_RANGE,
         In(Source, MapX, MapY), Dest, MapX.Step(), MapY.Step(), SrcImg, DstImg);
      break;
   case BestQuality:
   case Cubic:
      Kernel_(*m_CL, SelectProgram(Source), remap_cubic, Dest.FullRange(), LOCAL_RANGE,
         In(Source, MapX, MapY), Dest, MapX.Step(), MapY.Step(), SrcImg, DstImg);
      break;
   case Lanczos2:
   case Lanczos3:
   case SuperSampling:
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Rotate");
   }

}

void Transform::SetAll(Image& Dest, float Value)
{
   Kernel(set_all, In(Dest), Out(), Dest.Step(), Value);
}

void Transform::SetAll(Image& Dest, uint X, uint Y, uint Width, uint Height, float Value)
{
   Kernel(set_all_rect, In(Dest), Out(), Dest.Step(), X, Y, Width, Height, Value);
}

}
