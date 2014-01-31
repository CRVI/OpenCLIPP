////////////////////////////////////////////////////////////////////////////////
//! @file	: preprocessor.h 
//! @date   : Jul 2013
//!
//! @brief  : Useful preprocessor macros
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

/* This file provides useful preprocessor macros

CONCATENATE(a, b)             -> Outputs a concatenation of the two given arguments : CONCATENATE(ab, cd) -> abcd

STRIGIFY(a)                   -> Makes a string with the argument : STRIGIFY(a) -> "a"

REMOVE_PAREN(macro, arg)      -> Applies a list of parameters surrounded by parentheses to a macro : REMOVE_PAREN(a, (b, c)) -> a (b, c)

SELECT_FIRST(x, ...)          -> Selects the first argument of a list : SELECT_FIRST(a, b, c) -> a

VA_NUM_ARGS(...)              -> Outputs the number of aguments given : VA_NUM_ARGS(a, b, c) -> 3

HAS_ARGS(...)                 -> Outputs 1 if there are arguments present, outputs 0 if there are no arguments

ADD_COMMA(...)                -> Outputs a comma if it is given arguments, outputs nothing if there are no arguments

FOR_EACH(function, ...)       -> Applies function to each argument : FOR_EACH(test, a, b) -> test(a) test(b)

FOR_EACH_COMMA(function, ...) -> Generates a comma seperated list of each argument with 'function' applied to each
                                 FOR_EACH_COMMA(test, a, b, c) -> test(a), test(b), test(c)
                                 FOR_EACH_COMMA(test) ->       (nothing)
*/


/// Outputs a concatenation of the two given arguments : CONCATENATE(ab, cd) -> abcd
#define CONCATENATE(a, b) _CONCATENATE(a, b)

/// Makes a string with the argument : STRIGIFY(a) -> "a"
#define STRIGIFY(a) _STRIGIFY(a)
#define STR(a) STRIGIFY(a)       ///< Short version of STRIGIFY

/// Applies a list of parameters surrounded by parentheses to a macro : REMOVE_PAREN(a, (b, c)) -> a (b, c)
#define REMOVE_PAREN(macro, arg) macro arg

/// Selects the first given argument : SELECT_FIRST(a, b, c) -> a
#define SELECT_FIRST(x, ...) x

/// Outputs the number of arguments, outputs 0 if there are no arguments
#define VA_NUM_ARGS(...) __VA_NUM_ARGS(__VA_ARGS__)

/// Outputs 1 if there are arguments present, outputs 0 if there are no arguments
#define HAS_ARGS(...) __HAS_ARGS(__VA_ARGS__)

/// Outputs a comma if it is given arguments, outputs nothing if there are no arguments
#define ADD_COMMA(...) _ADD_COMMA(HAS_ARGS(__VA_ARGS__))

/// Applies function to each argument : FOR_EACH(test, a, b) -> test(a) test(b)
#define FOR_EACH(function, ...) _FOR_EACH_(VA_NUM_ARGS(__VA_ARGS__), function, __VA_ARGS__)

/// Generates a comma separated list of each argument with 'function' applied to each : FOR_EACH_COMMA(test, a, b, c) -> test(a), test(b), test(c)
#define FOR_EACH_COMMA(function, ...) _FOR_EACH_COMMA_(VA_NUM_ARGS(__VA_ARGS__), function, __VA_ARGS__)


// Implementation
#define _CONCATENATE(a, b) a ## b
#define _STRIGIFY(a) _STRIGIFY2(a)  // Two levels are needed for proper expansion of complex macros
#define _STRIGIFY2(a) #a
#define __ID(...) __VA_ARGS__

#ifdef _MSC_VER

// Works with Visual Studio 2012
#define __VA_NUM_ARGS(...) _VA_NUM_ARGS((0, __ID(__VA_ARGS__), 8, 7, 6, 5, 4, 3, 2, 1, 0))
#define __HAS_ARGS(...) _HAS_ARGS((0, __ID(__VA_ARGS__), 1, 1, 1, 1, 1, 1, 1, 1, 0))
#define _SELECT_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define _VA_NUM_ARGS(tuple) _SELECT_N tuple
#define _HAS_ARGS(tuple) _SELECT_N tuple

#else	// _MSC_VER


// Works with g++
#define _COMMA(...) ,
#define _HAS_COMMA(...) _SELECT_N(0, __VA_ARGS__, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1)
#define __VA_NUM_ARGS(...) _NUM_ARGS1(_HAS_COMMA(__VA_ARGS__), \
      _HAS_COMMA(_COMMA __VA_ARGS__ ()), \
      _SELECT_N(0, __VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1))

#define __HAS_ARGS(...) _NUM_ARGS1(_HAS_COMMA(__VA_ARGS__), \
      _HAS_COMMA(_COMMA __VA_ARGS__ ()), 1)

#define _NUM_ARGS1(a, b, N)    _NUM_ARGS2(a, b, N)
#define _NUM_ARGS2(a, b, N)    _NUM_ARGS3_ ## a ## b(N)
#define _NUM_ARGS3_01(N)    0
#define _NUM_ARGS3_00(N)    1
#define _NUM_ARGS3_11(N)    N

#define _SELECT_N(_0, _1, _2, _3, _4, _5, _6, _7, _8, N, ...) N

#endif	// _MSC_VER

#define _ADD_COMMA(N) CONCATENATE(_ADD_COMMA_, N)
#define _FOR_EACH_(N, fun, ...) CONCATENATE(_FOR_EACH_, N)(fun, __VA_ARGS__)
#define _FOR_EACH_COMMA_(N, fun, ...) CONCATENATE(_FOR_EACH_COMMA_, N)(fun, __VA_ARGS__)

#define _ADD_COMMA_0
#define _ADD_COMMA_1 , 

#define _FOR_EACH_0(...)
#define _FOR_EACH_1(x, a) x(a)
#define _FOR_EACH_2(x, a, b) x(a) _FOR_EACH_1(x, b)
#define _FOR_EACH_3(x, a, b, c) x(a) _FOR_EACH_2(x, b, c)
#define _FOR_EACH_4(x, a, b, c, d) x(a) _FOR_EACH_3(x, b, c, d)
#define _FOR_EACH_5(x, a, b, c, d, e) x(a) _FOR_EACH_4(x, b, c, d, e)
#define _FOR_EACH_6(x, a, b, c, d, e, f) x(a) _FOR_EACH_5(x, b, c, d, e, f)
#define _FOR_EACH_7(x, a, b, c, d, e, f, g) x(a) _FOR_EACH_6(x, b, c, d, e, f, g)
#define _FOR_EACH_8(x, a, b, c, d, e, f, g, h) x(a) _FOR_EACH_7(x, b, c, d, e, f, g, h)

#define _FOR_EACH_COMMA_0(...)
#define _FOR_EACH_COMMA_1(x, a) x(a)
#define _FOR_EACH_COMMA_2(x, a, b) _FOR_EACH_COMMA_1(x, a), x(b)
#define _FOR_EACH_COMMA_3(x, a, b, c) _FOR_EACH_COMMA_2(x, a, b), x(c)
#define _FOR_EACH_COMMA_4(x, a, b, c, d) _FOR_EACH_COMMA_3(x, a, b, c), x(d)
#define _FOR_EACH_COMMA_5(x, a, b, c, d, e) _FOR_EACH_COMMA_4(x, a, b, c, d), x(e)
#define _FOR_EACH_COMMA_6(x, a, b, c, d, e, f) _FOR_EACH_COMMA_5(x, a, b, c, d, e), x(f)
#define _FOR_EACH_COMMA_7(x, a, b, c, d, e, f, g) _FOR_EACH_COMMA_6(x, a, b, c, d, e, f), x(g)
#define _FOR_EACH_COMMA_8(x, a, b, c, d, e, f, g, h) _FOR_EACH_COMMA_7(x, a, b, c, d, e, f, g), x(h)
