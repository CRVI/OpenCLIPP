////////////////////////////////////////////////////////////////////////////////
//! @file	: TransformBuffer.cpp
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

#include "Programs/TransformBuffer.h"


#include "kernel_helpers.h"

// Transpose uses a local array and a local size of 32*8
#define PIXELS_PER_WORKITEM_V   4
#define LOCAL_WIDTH  32
#define LOCAL_HEIGHT 8

#include "WorkGroup.h"

#include <cmath>


static const double Pi = atan(1) * 4;


namespace OpenCLIPP
{

void TransformBuffer::MirrorX(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_x, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::MirrorY(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(mirror_y, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::Flip(ImageBuffer& Source, ImageBuffer& Dest)
{
   CheckSimilarity(Source, Dest);

   Kernel(flip, Source, Dest, Source.Step(), Dest.Step(), Source.Width(), Source.Height());
}

void TransformBuffer::Transpose(ImageBuffer& Source, ImageBuffer& Dest)
{
   if (Dest.Width() < Source.Height() || Dest.Height() < Source.Width())
      throw cl::Error(CL_INVALID_IMAGE_SIZE, "Destination image too small to receive TransformBuffer::Transpose");

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

void TransformBuffer::Rotate(ImageBuffer& Source, ImageBuffer& Dest,
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
   case SuperSampling:
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Rotate");
   }

}

void TransformBuffer::Resize(ImageBuffer& Source, ImageBuffer& Dest, EInterpolationType Interpolation, bool KeepRatio)
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
      Kernel_(*m_CL, SelectProgram(Source), resize_nn, Dest.FullRange(), LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case Linear:
      Kernel_(*m_CL, SelectProgram(Source), resize_linear, Dest.FullRange(), LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case Cubic:
      Kernel_(*m_CL, SelectProgram(Source), resize_bicubic, Dest.FullRange(), LOCAL_RANGE,
         In(Source), Out(Dest), SrcImg, DstImg, RatioX, RatioY);
      break;
   case BestQuality:
      if (RatioX < 1 || RatioY < 1)
      {
         // Enlarging image, SuperSampling is best
         //Resize(Source, Dest, SuperSampling, KeepRatio);
         Resize(Source, Dest, Cubic, KeepRatio);    // Use bicubic until SuperSampling is done
         break;
      }

      // Shrinking image, Linear is best
      Resize(Source, Dest, Linear, KeepRatio);
      break;
   case SuperSampling:
   default:
      throw cl::Error(CL_INVALID_ARG_VALUE, "Unsupported interpolation type in Rotate");
   }

}

void TransformBuffer::SetAll(ImageBuffer& Dest, float Value)
{
   Kernel(set_all, In(Dest), Out(), Dest.Step(), Value);
}

}
