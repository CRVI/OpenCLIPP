////////////////////////////////////////////////////////////////////////////////
//! @file	: Median.h
//! @date   : Mar 2014
//!
//! @brief  : Macros for median filters
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

// This file contains simple macros that are used to implement median filters

//The following macro puts the smallest value in position a and biggest in position b
#define s2(a, b)                Tmp = values[a]; values[a] = min(values[a], values[b]); values[b] = max(Tmp, values[b]);

//The following min macros make sure the first element is the minimum of a set (the remaining elements are in random order)
//The following max macros make sure the last element is the maximum of a set (the remaining elements are in random order)
#define min3(a, b, c)           s2(a, b); s2(a, c);
#define max3(a, b, c)           s2(b, c); s2(a, c);
#define min4(a,b,c,d)           min3(b, c, d); s2(a, b); 
#define max4(a,b,c,d)           max3(a, b, c); s2(c, d);
#define min5(a,b,c,d,e)         min4(b, c, d, e); s2(a, b); 
#define max5(a,b,c,d,e)         max4(a, b, c, d); s2(d, e);
#define min6(a,b,c,d,e,f)       min5(b, c, d, e, f); s2(a, b);
#define max6(a,b,c,d,e,f)       max5(a, b, c, d, e); s2(e, f);
#define min7(a,b,c,d,e,f,g)     min6(b, c, d, e, f, g); s2(a, b);
#define max7(a,b,c,d,e,f,g)     max6(a, b, c, d, e, f); s2(f, g);

//The following mnmx macros make sure the first element is the minimum and the last element is the maximum (the remaining elements are in random order)
#define mnmx3(a, b, c)          max3(a, b, c); s2(a, b);                                    // 3 exchanges
#define mnmx4(a, b, c, d)       s2(a, b); s2(c, d); s2(a, c); s2(b, d);                     // 4 exchanges
#define mnmx5(a, b, c, d, e)    s2(a, b); s2(c, d); min3(a, c, e); max3(b, d, e);           // 6 exchanges
#define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); min3(a, b, c); max3(d, e, f); // 7 exchanges
#define mnmx7(a,b,c,d,e,f,g)    s2(d,g); s2(a,e); s2(b,f); s2(c,g); min4(a,b,c,d); max3(e,f,g);
#define mnmx8(a,b,c,d,e,f,g,h)  s2(a,e); s2(b,f); s2(c,g); s2(d,h); min4(a,b,c,d); max4(e,f,g,h);
#define mnmx9(a,b,c,d,e,f,g,h,i) s2(e,i); s2(a,f); s2(b,g); s2(c,h); s2(d,i); min5(a,b,c,d,e); max4(f,g,h,i);
#define mnmx10(a,b,c,d,e,f,g,h,i,j) s2(a,f); s2(b,g); s2(c,h); s2(d,i); s2(e,j); min5(a,b,c,d,e); max5(f,g,h,i,j);
#define mnmx11(a,b,c,d,e,f,g,h,i,j,k) s2(f,k); s2(a,g); s2(b,h); s2(c,i); s2(d,j); s2(e,k); min6(a,b,c,d,e,f); max5(g,h,i,j,k);
#define mnmx12(a,b,c,d,e,f,g,h,i,j,k,l) s2(a,g); s2(b,h); s2(c,i); s2(d,j); s2(e,k); s2(f,l); min6(a,b,c,d,e,f); max6(g,h,i,j,k,l);
#define mnmx13(a,b,c,d,e,f,g,h,i,j,k,l,m)  s2(g,m); s2(a,h); s2(b,i); s2(c,j); s2(d,k); s2(e,l); s2(f,m); min7(a,b,c,d,e,f,g); max6(h,i,j,k,l,m);
#define mnmx14(a,b,c,d,e,f,g,h,i,j,k,l,m,n) s2(a,h); s2(b,i); s2(c,j); s2(d,k); s2(e,l); s2(f,m); s2(g,n); min7(a,b,c,d,e,f,g); max7(h,i,j,k,l,m,n);
