////////////////////////////////////////////////////////////////////////////////
//! @file	: FFT.h
//! @date   : Jan 2014
//!
//! @brief  : Fast Fourrier Transform using clFFT
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

#pragma once

#include "../Image.h"

typedef size_t clfftPlanHandle;

namespace OpenCLIPP
{

class CL_API FFT
{
public:
   FFT(COpenCL& CL);
   virtual ~FFT();

   /// Prepare processing for these images
   /// \param Real : Must be a 1 channel F32 image. Its size determines the problem size
   /// \param Complex : Must be a 2 channel F32 image. It must have the same height as Real and have a width that is : >=Real.Width()/2+1
   void PrepareFor(const ImageBase& Real, const ImageBase& Complex);  

   /// Performs a forward Fast Fourrier Transform.
   /// \param RealSource : Input image, must be 1 channel F32. Only real data, no imaginary component.
   /// \param ComplexDest : Output image, must be 2 channels F32.
   ///                      Channel 1 is the real component and channel 2 is the imaginary component.
   ///                      Width must be >=RealSource.Width()/2+1
   void Forward(Image& RealSource, Image& ComplexDest);

   /// Performs an inverse (backward) Fast Fourrier Transform.
   /// \param ComplexSource : Input image, must be 2 channels F32.
   ///                        Channel 1 is the real component and channel 2 is the imaginary component.
   ///                        Width must be >=RealDest.Width()/2+1
   /// \param RealDest : Output image, must be 1 channel F32. Only real data, no imaginary component.
   void Inverse(Image& ComplexSource, Image& RealDest);

   bool IsSupportedLength(uint Length) const;

protected:

   void ReleasePlans();

   COpenCL * m_CL;   ///< Pointer to the COpenCL object this program is assotiated to
   cl_command_queue m_Queue;

   bool IsPlanCompatible(clfftPlanHandle ForwardPlan, const ImageBase& Real, const ImageBase& Complex);
   
   clfftPlanHandle m_ForwardPlan;
   clfftPlanHandle m_BackwardPlan;

   std::shared_ptr<TempBuffer> m_ForwardTempBuffer;
   std::shared_ptr<TempBuffer> m_BackwardTempBuffer;

   bool m_OnNvidia;
};

}
