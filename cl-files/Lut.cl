////////////////////////////////////////////////////////////////////////////////
//! @file	: Lut.cl
//! @date   : Jul 2013
//!
//! @brief  : Lut transformation of images
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


SCALAR lut(SCALAR input, constant const uint * levels, constant const uint * values, int nb)
{
   if (input < levels[0])
      return input;

   if (input >= levels[nb - 1])
      return input;

   int k = 0;
   while (k < nb - 1 && input >= levels[k + 1])
      k++;

   return values[k];
}

SCALAR lut_linear(SCALAR input, constant const float * levels, constant const float * values, int nb)
{
   if (input < levels[0])
      return input;

   if (input >= levels[nb - 1])
      return input;

   int k = 0;
   while (k < nb - 1 && input > levels[k])
      k++;

   if (k > 0)
      k--;

   float DiffL = levels[k + 1] - levels[k];
   float DiffV = values[k + 1] - values[k];
   float Diff2 = input - levels[k];
   float Diff = Diff2 / DiffL * DiffV;

   return values[k] + Diff;
}

kernel void lut_1C(INPUT source, OUTPUT dest, constant const uint * levels, constant const uint * values, int nb)
{
   BEGIN

   TYPE color = READ_IMAGE(source, pos);

   color.x = lut(color.x, levels, values, nb);

   WRITE_IMAGE(dest, pos, color);
}

kernel void lut_4C(INPUT source, OUTPUT dest, constant const uint * levels, constant const uint * values, int nb)
{
   BEGIN

   TYPE color = READ_IMAGE(source, pos);

   color.x = lut(color.x, levels, values, nb);
   color.y = lut(color.y, levels, values, nb);
   color.z = lut(color.z, levels, values, nb);
   color.w = lut(color.w, levels, values, nb);

   WRITE_IMAGE(dest, pos, color);
}

kernel void lut_linear_1C(INPUT source, OUTPUT dest, constant const float * levels, constant const float * values, int nb)
{
   BEGIN

   TYPE color = READ_IMAGE(source, pos);

   color.x = lut_linear(color.x, levels, values, nb);

   WRITE_IMAGE(dest, pos, color);
}

kernel void lut_linear_4C(INPUT source, OUTPUT dest, constant const float * levels, constant const float * values, int nb)
{
   BEGIN

   TYPE color = READ_IMAGE(source, pos);

   color.x = lut_linear(color.x, levels, values, nb);
   color.y = lut_linear(color.y, levels, values, nb);
   color.z = lut_linear(color.z, levels, values, nb);
   color.w = lut_linear(color.w, levels, values, nb);

   WRITE_IMAGE(dest, pos, color);
}
