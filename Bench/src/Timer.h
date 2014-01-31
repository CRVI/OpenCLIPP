////////////////////////////////////////////////////////////////////////////////
//! @file	: Timer.h 
//! @date   : Jul 2013
//!
//! @brief  : Precise timer object for benchmarking
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

#pragma once


#ifndef _MSC_VER
#include <sys/time.h>
typedef timeval _time_type;
#else    // _MSC_VER
#include <windows.h>
typedef __int64 _time_type;
#endif   // _MSC_VER


class CTimer
{
public:
   CTimer();

   void Start();

   double Read();    ///< Read value in seconds

   double Readms()   ///< Read value in mili-seconds
   {
      return Read() * 1000;
   }

private:

   double m_Frequency;
   _time_type m_Startvalue;
};


#ifndef _MSC_VER


inline CTimer::CTimer()
{
   Start();
}

inline void CTimer::Start()
{
   gettimeofday(&m_Startvalue, nullptr);
}

inline double CTimer::Read()  ////< Read value in seconds
{
   _time_type End;
   gettimeofday(&End, nullptr);
   return End.tv_sec - m_Startvalue.tv_sec + double(End.tv_usec - m_Startvalue.tv_usec) / (1000 * 1000);
}

#else    // _MSC_VER

inline CTimer::CTimer()
{
   LARGE_INTEGER li;
   QueryPerformanceFrequency(&li);

   m_Frequency = double(li.QuadPart);

   Start();
}

inline void CTimer::Start()
{
   LARGE_INTEGER li;
   QueryPerformanceCounter(&li);
   m_Startvalue = li.QuadPart;
}

inline double CTimer::Read()  ////< Read value in seconds
{
   LARGE_INTEGER li;
   QueryPerformanceCounter(&li);
   return double(li.QuadPart - m_Startvalue) / m_Frequency;
}

#endif   // _MSC_VER
