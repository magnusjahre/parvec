/*
   SIMD Wrapper library by Juan M. Cebrian, NTNU - 2013.

   Version 0.1. ParVec compatibility.

   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.

*/


#ifndef __SIMD_DEFINES__
#define __SIMD_DEFINES__

//#define DEBUG

#include "icc_gcc_compat.h"
#include <stdio.h>
#include <math.h>
#include <stdint.h>

/* JMCG */

// JMCG Note: At this point in time NEON does not support double precission fp (Jun.2013)
#ifdef DFTYPE

#ifdef PARSEC_USE_SSE
#include <immintrin.h> // ALL SSE and AVX

#define _MM_ALIGNMENT 16
#define _MM_MANTISSA_BITS 52
#define _MM_MANTISSA_MASK 0x000fffffffffffffL
#define _MM_EXP_MASK 0x7ff0000000000000L
#define _MM_EXP_BIAS 0x00000000000003ffL
#define _MM_MINNORMPOS (1 << 20)
#define _MM_TYPE __m128d
#define _MM_TYPE_I __m128i
#define _MM_SCALAR_TYPE double
#define SIMD_WIDTH 2
#define _MM_SETZERO _mm_setzero_pd
#define _MM_SETZERO_I _mm_setzero_si128
#define _MM_ABS _mm_abs_pd
#define _MM_NEG _mm_neg_pd
#define _MM_LOG simd_log
#define _MM_EXP simd_exp
#define _MM_CMPGT _mm_cmpgt_pd
#define _MM_CMPLT _mm_cmplt_pd
#define _MM_CMPLE _mm_cmple_pd
#define _MM_CMPEQ _mm_cmpeq_pd
#define _MM_CMPEQ_SIG _mm_cmpeq_epi64
#define _MM_SRLI_I _mm_srli_epi64
#define _MM_SLLI_I _mm_slli_epi64
#define _MM_ADD_I _mm_add_epi64
#define _MM_SUB_I _mm_sub_epi64
#define _MM_CAST_FP_TO_I _mm_castpd_si128
#define _MM_CAST_I_TO_FP _mm_castsi128_pd
#define _MM_OR  _mm_or_pd
#define _MM_AND  _mm_and_pd
#define _MM_ANDNOT  _mm_andnot_pd
#define _MM_FLOOR _mm_floor_pd
#define _MM_LOAD  _mm_load_pd
#define _MM_LOADU  _mm_loadu_pd
#define _MM_LOAD3 _mm_load_st3
#define _MM_STORE _mm_store_pd
#define _MM_STOREU _mm_storeu_pd
#define _MM_MUL   _mm_mul_pd
#define _MM_ADD   _mm_add_pd
#define _MM_SUB   _mm_sub_pd
#define _MM_DIV   _mm_div_pd
#define _MM_SQRT  _mm_sqrt_pd
#define _MM_HADD _mm_hadd_pd
#define _MM_RHADD _mm_hadd_pd // JMCG REAL HADD, totally horizontal
#define _MM_FULL_HADD _mm_fullhadd_f64
#define _MM_CVT_F _mm_cvtsd_f64
#define _MM_CVT_I_TO_FP _mm_cvtepi64_pd
#define _MM_CVT_FP_TO_I _mm_cvtpd_epi64
#define _MM_SET(A)  _mm_set1_pd(A)
#define _MM_SETM(A,B)  _mm_set_pd(A,B)
#define _MM_SET_I(A)  _mm_set1_epi64x(A)
#define _MM_SETM_I(A,B)  _mm_set_epi64x(A,B)
#define _MM_SETR  _mm_setr_pd
#define _MM_MOVEMASK _mm_movemask_pd
#define _MM_MASK_TRUE 3 // 2 bits at 1
#define _MM_MAX _mm_max_pd
#define _MM_MIN _mm_min_pd
#define _MM_ATAN _mm_atan_pd
#define _MM_BLENDV _mm_blendv_pd
#define _MM_COPYSIGN _mm_copysign_pd // _MM_COPYSIGN(X,Y) takes sign from Y and copies it to X
#define _MM_MALLOC(A,B) _mm_malloc(A,B)

// Only for doubles, create code for floats
#define _MM_SHIFT_LEFT _mm_shift_left_pd
#define _MM_SHIFT2_LEFT _mm_shift2_left_pd
#define _MM_SHIFT_RIGHT _mm_shift_right_pd
#define _MM_SHIFT2_RIGHT _mm_shift2_right_pd
#define _MM_REVERSE _mm_reverse_pd
#define _MM_FMA _mm_fmadd_pd
#define _MM_PRINT_XMM print_xmm
#define _MM_PRINT_XMM_I print_xmm_i

#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (16)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

#if !defined (__AVX512VL__) && !defined (__AVX512DQ__)
static inline _MM_TYPE _mm_cvtepi64_pd(_MM_TYPE_I A) {
  return _mm_cvtepi32_pd(_mm_shuffle_epi32(A, _MM_SHUFFLE (3, 3, 2, 0)));
}
static inline _MM_TYPE_I _mm_cvtpd_epi64(_MM_TYPE A) {
  return _mm_cvtepi32_epi64(_mm_cvtpd_epi32(A));
}
#endif

// For debugging
static inline void print_xmm(_MM_TYPE in, char* s) {
  int i;
  _MM_ALIGN double val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%.16f ", val[i]);
  }
  printf("\n");
}

static inline void print_xmm_i(_MM_TYPE_I in, char* s) {
  int i;
  int64_t val[SIMD_WIDTH];

  _mm_storeu_si128((__m128i*)&val[0], in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%ld ", val[i]);
  }
  printf("\n");
}



__attribute__((aligned (16))) static const int absmask_double[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
#define _mm_abs_pd(x) _mm_and_pd((x), *(const __m128d*)absmask_double)
__attribute__((aligned (16))) static const int negmask_double[] = { 0xffffffff, 0x80000000, 0xffffffff, 0x80000000};
#define _mm_neg_pd(x) _mm_xor_pd((x), *(const __m128d*)negmask_double)

static inline _MM_TYPE _mm_copysign_pd(_MM_TYPE x, _MM_TYPE y) {
  return _mm_or_pd(_MM_ABS(x),_mm_and_pd(y,*(const __m128d*)negmask_double));
}

#ifndef __FMA3__
#ifndef _FMAINTRIN_H_INCLUDED
static inline _MM_TYPE _mm_fmadd_pd(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return _MM_ADD(_MM_MUL(A,B),C);
}
#endif
#endif

static inline _MM_TYPE _mm_reverse_pd(_MM_TYPE A) {
  return _mm_shuffle_pd(A,A,_MM_SHUFFLE2(0,1)); // return = {A[1],A[0]}
}

static inline _MM_TYPE _mm_shift_left_pd(_MM_TYPE A) {
  return _mm_shuffle_pd(A,A,_MM_SHUFFLE2(0,0)); // return = {A[0],A[0]}
}

static inline _MM_TYPE _mm_shift2_left_pd(_MM_TYPE A, _MM_TYPE B) {
  // B comes reversed to make it uniform with AVX version
  return _mm_shuffle_pd(B,A,_MM_SHUFFLE2(0,0)); // return = {B[0],A[0]}
}

static inline _MM_TYPE _mm_shift_right_pd(_MM_TYPE A) {
  return _mm_shuffle_pd(A,A,_MM_SHUFFLE2(1,1)); // return = {A[1],A[1]}
}

static inline _MM_TYPE _mm_shift2_right_pd(_MM_TYPE A, _MM_TYPE B, _MM_TYPE *C) {
  // We return reverse A to make it uniform with AVX version
  *C = _mm_reverse_pd(A);
  return _mm_shuffle_pd(A,B, _MM_SHUFFLE2(0,1)); // return = {A[1],B[0]}
}

static inline _MM_TYPE _mm_fullhadd_f64(_MM_TYPE A, _MM_TYPE B) {
  return _mm_hadd_pd(A,B);
}

static inline void _mm_load_st3(_MM_TYPE* A, _MM_TYPE* B, _MM_TYPE* C, double* address) {
  _MM_TYPE a_temp, b_temp, c_temp;

  a_temp = _MM_LOAD(address); // b1,a1
  b_temp = _MM_LOAD(address+2); // a2,c1
  c_temp = _MM_LOAD(address+4); // c2,b2
  *A = _mm_blend_pd(a_temp,b_temp,0b10); // A = a2,a1
  *B = _mm_shuffle_pd(a_temp,c_temp, _MM_SHUFFLE2(0,1)); // B = b2,b1
  *C = _mm_blend_pd(b_temp,c_temp,0b10); // C = c2,c1
}

// Algorithm taken from vecmathlib and SLEEF 2.80
static inline _MM_TYPE _mm_atan_pd(_MM_TYPE A) {
  _MM_TYPE q1 = A;
  _MM_TYPE s = _MM_ABS(A);

  _MM_TYPE q0 = _MM_CMPGT(s,_MM_SET(1.0));
  s = _MM_BLENDV(s,_MM_DIV(_MM_SET(1.0),s),q0); // s = ifthen(q0, rcp(s), s);

  _MM_TYPE t = _MM_MUL(s,s); //  realvec_t t = s * s;
  _MM_TYPE u = _MM_SET(-1.88796008463073496563746e-05);

  u = _MM_FMA(u,t,_MM_SET(0.000209850076645816976906797));
  u = _MM_FMA(u,t,_MM_SET(-0.00110611831486672482563471));
  u = _MM_FMA(u,t,_MM_SET(0.00370026744188713119232403));
  u = _MM_FMA(u,t,_MM_SET(-0.00889896195887655491740809));
  u = _MM_FMA(u,t,_MM_SET(0.016599329773529201970117));
  u = _MM_FMA(u,t,_MM_SET(-0.0254517624932312641616861));
  u = _MM_FMA(u,t,_MM_SET(0.0337852580001353069993897));
  u = _MM_FMA(u,t,_MM_SET(-0.0407629191276836500001934));
  u = _MM_FMA(u,t,_MM_SET(0.0466667150077840625632675));
  u = _MM_FMA(u,t,_MM_SET(-0.0523674852303482457616113));
  u = _MM_FMA(u,t,_MM_SET(0.0587666392926673580854313));
  u = _MM_FMA(u,t,_MM_SET(-0.0666573579361080525984562));
  u = _MM_FMA(u,t,_MM_SET(0.0769219538311769618355029));
  u = _MM_FMA(u,t,_MM_SET(-0.090908995008245008229153));
  u = _MM_FMA(u,t,_MM_SET(0.111111105648261418443745));
  u = _MM_FMA(u,t,_MM_SET(-0.14285714266771329383765));
  u = _MM_FMA(u,t,_MM_SET(0.199999999996591265594148));
  u = _MM_FMA(u,t,_MM_SET(-0.333333333333311110369124));

  t = _MM_ADD(s,_MM_MUL(s,_MM_MUL(t,u))); //  t = s + s * (t * u);
  t = _MM_BLENDV(t,_MM_SUB(_MM_SET(M_PI_2),t),q0); // t = ifthen(q0, RV(M_PI_2) - t, t);
  t = _MM_COPYSIGN(t, q1);

  return t;
}

#endif // PARSEC_USE_SSE

#ifdef PARSEC_USE_AVX
#include <immintrin.h> // ALL SSE and AVX

#define _MM_ALIGNMENT 32
#define _MM_MANTISSA_BITS 52
#define _MM_MANTISSA_MASK 0x000fffffffffffffL
#define _MM_EXP_MASK 0x7ff0000000000000L
#define _MM_EXP_BIAS 0x00000000000003ffL
#define _MM_MINNORMPOS (1 << 20)
#define _MM_TYPE  __m256d
#define _MM_TYPE_I __m256i
#define _MM_SCALAR_TYPE double
#define SIMD_WIDTH 4
#define _MM_SETZERO _mm256_setzero_pd
#define _MM_SETZERO_I _mm256_setzero_si256
#define _MM_ABS _mm256_abs_pd
#define _MM_NEG _mm256_neg_pd
#define _MM_LOG simd_log
#define _MM_EXP simd_exp
#define _MM_CMPGT _mm256_cmpgt_pd
#define _MM_CMPLT _mm256_cmplt_pd
#define _MM_CMPLE _mm256_cmple_pd
#define _MM_CMPEQ _mm256_cmpeq_pd

#ifndef __AVX2__ // GCC 5+ ALWAYS define cmpeq even if AVX2 is not available, so we can no longer implement our own with the same name
#define _MM_CMPEQ_SIG _custom_mm256_cmpeq_epi64
#else
#define _MM_CMPEQ_SIG _mm256_cmpeq_epi64
#endif

#define _MM_SRLI_I _mm256_srli_epi64
#define _MM_SLLI_I _mm256_slli_epi64
#define _MM_ADD_I _mm256_add_epi64
#define _MM_SUB_I _mm256_sub_epi64
#define _MM_CAST_FP_TO_I _mm256_castpd_si256
#define _MM_CAST_I_TO_FP _mm256_castsi256_pd
#define _MM_OR  _mm256_or_pd
#define _MM_AND  _mm256_and_pd
#define _MM_ANDNOT  _mm256_andnot_pd
#define _MM_FLOOR _mm256_floor_pd
#define _MM_LOAD  _mm256_load_pd
#define _MM_LOADU  _mm256_loadu_pd
#define _MM_LOAD3 _mm_load_st3
#define _MM_STORE _mm256_store_pd
#define _MM_STOREU _mm256_storeu_pd
#define _MM_MUL   _mm256_mul_pd
#define _MM_ADD   _mm256_add_pd
#define _MM_SUB   _mm256_sub_pd
#define _MM_DIV   _mm256_div_pd
#define _MM_SQRT  _mm256_sqrt_pd
#define _MM_HADD _mm256_hadd_pd
#define _MM_RHADD _mm256_rhadd_pd // JMCG REAL HADD, totally horizontal
#define _MM_FULL_HADD _mm256_fullhadd_f64
#define _MM_CVT_F _mm_cvtsd_f64
#define _MM_CVT_I_TO_FP _mm256_cvtepi64_pd
#define _MM_CVT_FP_TO_I _mm256_cvtpd_epi64
#define _MM_SET(A)  _mm256_set1_pd(A)
#define _MM_SETM(A,B,C,D)  _mm256_set_pd(A,B,C,D)
#define _MM_SET_I(A)  _mm256_set1_epi64x(A)
#define _MM_SETM_I(A,B,C,D)  _mm256_set_epi64x(A,B,C,D)
#define _MM_SETR  _mm256_setr_pd
#define _MM_BROADCAST  _mm256_broadcast_pd
#define _MM_MOVEMASK _mm256_movemask_pd
#define _MM_MASK_TRUE 15 // 4 bits at 1
#define _MM_MAX _mm256_max_pd
#define _MM_MIN _mm256_min_pd
#define _MM_ATAN _mm256_atan_pd
#define _MM_BLENDV _mm256_blendv_pd
#define _MM_COPYSIGN _mm256_copysign_pd // _MM_COPYSIGN(X,Y) takes sign from Y and copies it to X
#define _MM_MALLOC(A,B) _mm_malloc(A,B)
#define _MM_FMA _mm256_fmadd_pd
#define _MM_PRINT_XMM print_xmm
#define _MM_PRINT_XMM_I print_xmm_i

// Only for doubles, create code for floats
#define _MM_SHIFT_LEFT _mm256_shift_left_pd
#define _MM_SHIFT2_LEFT _mm256_shift2_left_pd
#define _MM_SHIFT_RIGHT _mm256_shift_right_pd
#define _MM_SHIFT2_RIGHT _mm256_shift2_right_pd
#define _MM_REVERSE _mm256_reverse_pd


#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (32)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

#if !defined (__AVX512VL__) && !defined (__AVX512DQ__)
static inline _MM_TYPE _mm256_cvtepi64_pd(_MM_TYPE_I x) {
  _MM_TYPE output;
  __m128i emm01 = _mm_shuffle_epi32(_mm256_extractf128_si256(x, 0), _MM_SHUFFLE (3, 3, 2, 0));
  __m128i emm02 = _mm_shuffle_epi32(_mm256_extractf128_si256(x, 1), _MM_SHUFFLE (3, 3, 2, 0));
  output = _mm256_insertf128_pd(output, _mm_cvtepi32_pd(emm01), 0);
  output = _mm256_insertf128_pd(output, _mm_cvtepi32_pd(emm02), 1);
  return output;
}
static inline _MM_TYPE_I _mm256_cvtpd_epi64(_MM_TYPE x) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_cvtpd_epi32(_mm256_extractf128_pd(x, 0));
  __m128i emm02 = _mm_cvtpd_epi32(_mm256_extractf128_pd(x, 1));
  output = _mm256_insertf128_si256(output, _mm_cvtepi32_epi64(emm01), 0);
  output = _mm256_insertf128_si256(output, _mm_cvtepi32_epi64(emm02), 1);
  return output;
}
#endif

#ifndef __AVX2__
static inline _MM_TYPE_I _mm256_srli_epi64(_MM_TYPE_I x, uint32_t imm8) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_srli_epi64(_mm256_extractf128_si256(x, 0), imm8);
  __m128i emm02 = _mm_srli_epi64(_mm256_extractf128_si256(x, 1), imm8);
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_slli_epi64(_MM_TYPE_I x, uint32_t imm8) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_slli_epi64(_mm256_extractf128_si256(x, 0), imm8);
  __m128i emm02 = _mm_slli_epi64(_mm256_extractf128_si256(x, 1), imm8);
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_add_epi64(_MM_TYPE_I x, _MM_TYPE_I y) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_add_epi64(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(y, 0));
  __m128i emm02 = _mm_add_epi64(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_sub_epi64(_MM_TYPE_I x, _MM_TYPE_I y) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_sub_epi64(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(y, 0));
  __m128i emm02 = _mm_sub_epi64(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
#endif

#ifndef __FMA3__
#ifndef _FMAINTRIN_H_INCLUDED
static inline _MM_TYPE _mm256_fmadd_pd(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return _MM_ADD(_MM_MUL(A,B),C);
}
#endif
#endif

__attribute__((aligned (32))) static const int absmask_double_256[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
#define _mm256_abs_pd(x) _mm256_and_pd((x), *(const __m256d*)absmask_double_256)
__attribute__((aligned (32))) static const int negmask_double_256[] = { 0xffffffff, 0x80000000, 0xffffffff, 0x80000000, 0xffffffff, 0x80000000, 0xffffffff, 0x80000000};
#define _mm256_neg_pd(x) _mm256_xor_pd((x), *(const __m256d*)negmask_double_256)

static inline _MM_TYPE _mm256_copysign_pd(_MM_TYPE x, _MM_TYPE y) {
  return _mm256_or_pd(_MM_ABS(x),_mm256_and_pd(y,*(const __m256d*)negmask_double_256));
}

// For debugging
static inline void print_xmm(_MM_TYPE in, char* s) {
  int i;
  _MM_ALIGN double val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%.16f ", val[i]);
  }
  printf("\n");
}

static inline void print_xmm_i(_MM_TYPE_I in, char* s) {
  int i;
  int64_t val[SIMD_WIDTH];

  _mm256_storeu_si256((__m256i*)&val[0], in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%ld ", val[i]);
  }
  printf("\n");
}



static inline _MM_TYPE _mm256_reverse_pd(_MM_TYPE A) {
  _MM_TYPE temp;
  temp = _mm256_permute_pd( A, 0b0101); // temp = {A[1],A[0],A[3],A[2]}
  return (_mm256_permute2f128_pd( temp, temp, _MM_SHUFFLE2(0,1))); // temp4 = {A[3],A[2],A[1],A[0]}
}

static inline _MM_TYPE _mm256_shift_left_pd(_MM_TYPE A) {
  _MM_TYPE temp, temp2, temp3;
  temp = _mm256_permute_pd( A, 0b0001); // temp = {A[1],A[0],A[2],A[2]}
  temp2 =  _mm256_permute2f128_pd( temp, temp, _MM_SHUFFLE2(0,1)); // temp2 = {A[2],A[2],A[1],A[0]}
  temp3 = _mm256_blend_pd( temp, temp2, 0b0101); // temp3 = {A[2],A[0],A[1],A[2]}
  return (_mm256_blend_pd( temp3, A, 0b0001)); // return = {A[0],A[0],A[1],A[2]}
}

static inline _MM_TYPE _mm256_shift2_left_pd(_MM_TYPE A, _MM_TYPE B) {
  _MM_TYPE temp, temp2, temp3;
  temp = _mm256_permute_pd( A, 0b0001); // temp = {A[1], A[0], A[2], A[2]}
  temp2 = _mm256_permute2f128_pd( temp, temp, _MM_SHUFFLE2(0,1)); // temp2 = {A[2], A[2], A[1], A[0]}
  temp3 = _mm256_blend_pd(temp, temp2, 0b0101); // temp3 = {A[2], A[0], A[1], A[2]}
  return (_mm256_blend_pd( temp3, B, 0b0001)); // return = {B[0],A[0],A[1],A[2]}
}

static inline _MM_TYPE _mm256_shift_right_pd(_MM_TYPE A) {
  _MM_TYPE temp, temp2;
  temp = _mm256_permute_pd( A, 0b1101); // temp = {A[1],A[0],A[3],A[3]}
  temp2 = _mm256_permute_pd( A, 0b0001); // temp = {A[1],A[0],A[2],A[2]}
  temp2 =  _mm256_permute2f128_pd( temp2, temp2, _MM_SHUFFLE2(0,1)); // temp2 = {A[2],A[2],A[1],A[0]}
  return (_mm256_blend_pd( temp, temp2, 0b0010)); // return = {A[0],A[0],A[1],A[2]}
}

static inline _MM_TYPE _mm256_shift2_right_pd(_MM_TYPE A, _MM_TYPE B, _MM_TYPE* C) {
  // Also returns reversed A in C
  _MM_TYPE temp2, temp3, temp4, temp5;
  temp2 = _mm256_permute2f128_pd( B, B, _MM_SHUFFLE2(0,1)); // temp2 = {B[2],B[3],B[0],B[1]}
  temp3 = _mm256_permute_pd( A, 0b0101); // temp3 = {A[1],A[0],A[3],A[2]}
  temp4 = _mm256_permute2f128_pd( temp3, temp3, _MM_SHUFFLE2(0,1)); // temp4 = {A[3],A[2],A[1],A[0]}
  *C = temp4;
  temp5 = _mm256_shuffle_pd(temp3, temp2, 0b0000); // temp5 = {A[1],B[3],A[3],B[0]}
  return (_mm256_blend_pd( temp5, temp4, 0b0010)); // return = {A[1],A[2],A[3],B[0]}
}

static inline _MM_TYPE_I _custom_mm256_cmpeq_epi64(_MM_TYPE_I A, _MM_TYPE_I B) {
  return (_MM_TYPE_I)_MM_CMPEQ((_MM_TYPE)A,(_MM_TYPE)B);
}

static inline _MM_TYPE _mm256_rhadd_pd(_MM_TYPE A, _MM_TYPE B) {

  _MM_TYPE temp = _mm256_hadd_pd(A,B); // B3+B4 A3+A4 B1+B2 A1+A2
  _MM_TYPE temp2 = _mm256_permute2f128_pd( temp , temp , _MM_SHUFFLE2(0,1)); // B1+B2 A1+A2 B3+B4 A3+A4
  _MM_TYPE temp3 = _mm256_shuffle_pd(temp,temp2,0b1100); // B1+B2 B3+B4 A3+A4 A1+A2

  return _mm256_shuffle_pd(temp3,temp3,0b0110); // B3+B4 B1+B2 A3+A4 A1+A2
}


static inline __m128d _mm256_fullhadd_f64(_MM_TYPE A, _MM_TYPE B) {
  __m128d  hi,lo;
  _MM_TYPE temp = _mm256_hadd_pd(A,B);

  hi = _mm256_extractf128_pd(temp,1);
  lo = _mm256_extractf128_pd(temp,0);

  return _mm_add_pd(hi,lo);
}

static inline double _mm256_cvtsd_f64(_MM_TYPE A) {
  return (_mm_cvtsd_f64(_mm256_extractf128_pd(A,0)));
}

static inline void _mm_load_st3(_MM_TYPE* A, _MM_TYPE* B, _MM_TYPE* C, double* address) {
  _MM_TYPE a_temp, b_temp, c_temp;
  _MM_TYPE aa, bs, bp, cc, cp;

  a_temp = _MM_LOAD(address); // a2,c1,b1,a1
  b_temp = _MM_LOAD(address+4); // b3,a3,c2,b2
  c_temp = _MM_LOAD(address+8); // c4,b4,a4,c3

  // reorder for A
  aa = _mm256_blend_pd(a_temp,b_temp,0b1110); // aa = b3,a3,c2,a1

  // reorder for B
  bp = _mm256_permute_pd(b_temp,0b0101); // bp = a3,b3,b2,c2
  bs = _mm256_shuffle_pd(a_temp,c_temp,0b0001); // bs = b4,c1,c3,b1

  // reorder for C
  cp = _mm256_blend_pd(c_temp,b_temp,0b0011); // cp = c4,b4,c2,b2

  // reorder for A & C
  cc = _mm256_permute2f128_pd(a_temp,c_temp,0b00100001); // aa = a4,c3,a2,c1

  *A = _mm256_blend_pd(aa,cc,0b1010); // A = a4,a3,a2,a1
  *B = _mm256_blend_pd(bs,bp,0b0110); // B = b4,b3,b2,b1
  *C = _mm256_blend_pd(cc,cp,0b1010); // C = c4,c3,c2,c1
}

// Algorithm taken from vecmathlib and SLEEF 2.80
static inline _MM_TYPE _mm256_atan_pd(_MM_TYPE A) {

  _MM_TYPE q1 = A;
  _MM_TYPE s = _MM_ABS(A);

  _MM_TYPE q0 = _MM_CMPGT(s,_MM_SET(1.0));
  s = _MM_BLENDV(s,_MM_DIV(_MM_SET(1.0),s),q0); // s = ifthen(q0, rcp(s), s);

  _MM_TYPE t = _MM_MUL(s,s); //  realvec_t t = s * s;
  _MM_TYPE u = _MM_SET(-1.88796008463073496563746e-05);

  u = _MM_FMA(u,t,_MM_SET(0.000209850076645816976906797));
  u = _MM_FMA(u,t,_MM_SET(-0.00110611831486672482563471));
  u = _MM_FMA(u,t,_MM_SET(0.00370026744188713119232403));
  u = _MM_FMA(u,t,_MM_SET(-0.00889896195887655491740809));
  u = _MM_FMA(u,t,_MM_SET(0.016599329773529201970117));
  u = _MM_FMA(u,t,_MM_SET(-0.0254517624932312641616861));
  u = _MM_FMA(u,t,_MM_SET(0.0337852580001353069993897));
  u = _MM_FMA(u,t,_MM_SET(-0.0407629191276836500001934));
  u = _MM_FMA(u,t,_MM_SET(0.0466667150077840625632675));
  u = _MM_FMA(u,t,_MM_SET(-0.0523674852303482457616113));
  u = _MM_FMA(u,t,_MM_SET(0.0587666392926673580854313));
  u = _MM_FMA(u,t,_MM_SET(-0.0666573579361080525984562));
  u = _MM_FMA(u,t,_MM_SET(0.0769219538311769618355029));
  u = _MM_FMA(u,t,_MM_SET(-0.090908995008245008229153));
  u = _MM_FMA(u,t,_MM_SET(0.111111105648261418443745));
  u = _MM_FMA(u,t,_MM_SET(-0.14285714266771329383765));
  u = _MM_FMA(u,t,_MM_SET(0.199999999996591265594148));
  u = _MM_FMA(u,t,_MM_SET(-0.333333333333311110369124));

  t = _MM_ADD(s,_MM_MUL(s,_MM_MUL(t,u))); //  t = s + s * (t * u);

  t = _MM_BLENDV(t,_MM_SUB(_MM_SET(M_PI_2),t),q0); // t = ifthen(q0, RV(M_PI_2) - t, t);
  t = _MM_COPYSIGN(t, q1);

  return t;
}

#endif // PARSEC_USE_AVX

#else // ~DFTYPE

#ifdef PARSEC_USE_SSE
//#include <xmmintrin.h> // SSE
//#include <emmintrin.h> // SSE 2
//#include <pmmintrin.h> // SSE 3
//#include <smmintrin.h> // SSE 4.1
#include <immintrin.h> // ALL SSE and AVX

#define _MM_ALIGNMENT 16
#define _MM_MANTISSA_BITS 23
#define _MM_MANTISSA_MASK 0x007fffff
#define _MM_EXP_MASK 0x7f800000
#define _MM_EXP_BIAS 0x7f
#define _MM_MINNORMPOS (1 << _MM_MANTISSA_BITS)
#define _MM_TYPE  __m128
#define _MM_TYPE_I __m128i
#define _MM_SCALAR_TYPE float
#define SIMD_WIDTH 4
#define _MM_SETZERO _mm_setzero_ps
#define _MM_SETZERO_I _mm_setzero_si128
#define _MM_ABS _mm_abs_ps
#define _MM_NEG _mm_neg_ps
#define _MM_LOG simd_log
#define _MM_EXP simd_exp
#define _MM_CMPGT _mm_cmpgt_ps
#define _MM_CMPLT _mm_cmplt_ps
#define _MM_CMPLE _mm_cmple_ps
#define _MM_CMPEQ _mm_cmpeq_ps
#define _MM_CMPEQ_SIG _mm_cmpeq_epi32
#define _MM_SRLI_I _mm_srli_epi32
#define _MM_SLLI_I _mm_slli_epi32
#define _MM_ADD_I _mm_add_epi32
#define _MM_SUB_I _mm_sub_epi32
#define _MM_CAST_FP_TO_I _mm_castps_si128
#define _MM_CAST_I_TO_FP _mm_castsi128_ps
#define _MM_OR  _mm_or_ps
#define _MM_AND  _mm_and_ps
#define _MM_ANDNOT  _mm_andnot_ps
#define _MM_FLOOR _mm_floor_ps
#define _MM_LOAD  _mm_load_ps
#define _MM_LOADU  _mm_loadu_ps
#define _MM_LOAD3 _mm_load_st3
#define _MM_STORE _mm_store_ps
#define _MM_STOREU _mm_storeu_ps
#define _MM_MUL   _mm_mul_ps
#define _MM_ADD   _mm_add_ps
#define _MM_SUB   _mm_sub_ps
#define _MM_DIV   _mm_div_ps
#define _MM_SQRT  _mm_sqrt_ps
#define _MM_HADD _mm_hadd_ps
#define _MM_RHADD _mm_hadd_ps // JMCG REAL HADD, totally horizontal
#define _MM_FULL_HADD _mm_fullhadd_f32
#define _MM_CVT_F _mm_cvtss_f32
#define _MM_CVT_I_TO_FP _mm_cvtepi32_ps
#define _MM_CVT_FP_TO_I _mm_cvtps_epi32
#define _MM_SET(A)  _mm_set1_ps(A)
#define _MM_SETM(A,B,C,D)  _mm_set_ps(A,B,C,D)
#define _MM_SET_I(A)  _mm_set1_epi32(A)
#define _MM_SETM_I(A,B,C,D)  _mm_set_epi32(A,B,C,D)
#define _MM_SETR  _mm_setr_ps
#define _MM_MOVEMASK _mm_movemask_ps
#define _MM_MASK_TRUE 15 // 4 bits at 1
#define _MM_MAX _mm_max_ps
#define _MM_MIN _mm_min_ps
#define _MM_ATAN _mm_atan_ps
#define _MM_BLENDV _mm_blendv_ps
#define _MM_COPYSIGN _mm_copysign_ps // _MM_COPYSIGN(X,Y) takes sign from Y and copies it to X
#define _MM_MALLOC(A,B) _mm_malloc(A,B)
#define _MM_FMA _mm_fmadd_ps
#define _MM_PRINT_XMM print_xmm
#define _MM_PRINT_XMM_I print_xmm_i

#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (16)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

#ifndef __FMA3__
#ifndef _FMAINTRIN_H_INCLUDED
static inline _MM_TYPE _mm_fmadd_ps(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return _MM_ADD(_MM_MUL(A,B),C);
}
#endif
#endif

__attribute__((aligned (16))) static const int absmask[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
#define _mm_abs_ps(x) _mm_and_ps((x), *(const __m128*)absmask)
__attribute__((aligned (16))) static const int negmask[] = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
#define _mm_neg_ps(x) _mm_xor_ps((x), *(const __m128*)negmask)

static inline _MM_TYPE _mm_copysign_ps(_MM_TYPE x, _MM_TYPE y) {
  return _mm_or_ps(_MM_ABS(x),_mm_and_ps(y,*(const __m128*)negmask));
}

// For debugging
static inline void print_xmm(_MM_TYPE in, char* s) {
  int i;
  _MM_ALIGN float val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%.16f ", val[i]);
  }
  printf("\n");
}

static inline void print_xmm_i(_MM_TYPE_I in, char* s) {
  int i;
  int32_t val[SIMD_WIDTH];

  _mm_storeu_si128((__m128i*)&val[0], in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%d ", val[i]);
  }
  printf("\n");
}


static inline _MM_TYPE _mm_fullhadd_f32(_MM_TYPE A, _MM_TYPE B) {
  _MM_TYPE temp = _mm_hadd_ps(A,B);
  return _mm_hadd_ps(temp,temp);
}


static inline void _mm_load_st3(_MM_TYPE* A, _MM_TYPE* B, _MM_TYPE* C, float* address) {
  _MM_TYPE a_temp, b_temp, c_temp;
  _MM_TYPE aa, bb, cc;

  a_temp = _MM_LOAD(address); // a2,c1,b1,a1
  b_temp = _MM_LOAD(address+4); // b3,a3,c2,b2
  c_temp = _MM_LOAD(address+8); // c4,b4,a4,c3

  aa = _mm_shuffle_ps(b_temp,c_temp,_MM_SHUFFLE(1,1,2,2)); // aa = a4,a4,a3,a3
  bb = _mm_shuffle_ps(a_temp,c_temp,_MM_SHUFFLE(2,2,1,1)); // bb = b4,b4,b1,b1
  cc = _mm_shuffle_ps(a_temp,b_temp,_MM_SHUFFLE(1,1,2,2)); // cc = c2,c2,c1,c1

  *A = _mm_shuffle_ps(a_temp,aa,_MM_SHUFFLE(2,1,3,0)); // A = a4,a3,a2,a1
  *C = _mm_shuffle_ps(cc,c_temp,_MM_SHUFFLE(3,0,2,1)); // C = c4,c3,c2,c1

  bb = _mm_blend_ps(b_temp,bb,0b0110);  // bb = b3,b4,b1,b2
  *B = _mm_shuffle_ps(bb,bb,_MM_SHUFFLE(2,3,0,1)); // B = b4,b3,b2,b1
}

// Algorithm taken from vecmathlib and SLEEF 2.80
static inline _MM_TYPE _mm_atan_ps(_MM_TYPE A) {
  _MM_TYPE q1 = A;
  _MM_TYPE s = _MM_ABS(A);

  _MM_TYPE q0 = _MM_CMPGT(s,_MM_SET(1.0));
  s = _MM_BLENDV(s,_MM_DIV(_MM_SET(1.0),s),q0); // s = ifthen(q0, rcp(s), s);

  _MM_TYPE t = _MM_MUL(s,s); //  realvec_t t = s * s;
  _MM_TYPE u = _MM_SET(0.00282363896258175373077393f);

  u = _MM_FMA(u,t,_MM_SET(-0.0159569028764963150024414f));
  u = _MM_FMA(u,t,_MM_SET(0.0425049886107444763183594f));
  u = _MM_FMA(u,t,_MM_SET(-0.0748900920152664184570312f));
  u = _MM_FMA(u,t,_MM_SET(0.106347933411598205566406f));
  u = _MM_FMA(u,t,_MM_SET(-0.142027363181114196777344f));
  u = _MM_FMA(u,t,_MM_SET(0.199926957488059997558594f));
  u = _MM_FMA(u,t,_MM_SET(-0.333331018686294555664062f));

  t = _MM_ADD(s,_MM_MUL(s,_MM_MUL(t,u))); //  t = s + s * (t * u);

  t = _MM_BLENDV(t,_MM_SUB(_MM_SET(M_PI_2),t),q0); // t = ifthen(q0, RV(M_PI_2) - t, t);
  t = _MM_COPYSIGN(t, q1);

  return t;
}

#endif  // PARSEC_USE_SSE

#ifdef PARSEC_USE_AVX
#include <immintrin.h> // ALL SSE and AVX

#define _MM_ALIGNMENT 32
#define _MM_MANTISSA_BITS 23
#define _MM_MANTISSA_MASK 0x007fffff
#define _MM_EXP_MASK 0x7f800000
#define _MM_EXP_BIAS 0x7f
#define _MM_MINNORMPOS (1 << _MM_MANTISSA_BITS)
#define _MM_TYPE  __m256
#define _MM_TYPE_I __m256i
#define _MM_SCALAR_TYPE float
#define SIMD_WIDTH 8
#define _MM_SETZERO _mm256_setzero_ps
#define _MM_SETZERO_I _mm256_setzero_si256
#define _MM_ABS _mm256_abs_ps
#define _MM_NEG _mm256_neg_ps
#define _MM_LOG simd_log
#define _MM_EXP simd_exp
#define _MM_CMPGT _mm256_cmpgt_ps
#define _MM_CMPLT _mm256_cmplt_ps
#define _MM_CMPLE _mm256_cmple_ps
#define _MM_CMPEQ _mm256_cmpeq_ps

#ifndef __AVX2__ // GCC 5+ ALWAYS define cmpeq even if AVX2 is not available, so we can no longer implement our own with the same name
#define _MM_CMPEQ_SIG _custom_mm256_cmpeq_epi32
#else
#define _MM_CMPEQ_SIG _mm256_cmpeq_epi32
#endif

#define _MM_SRLI_I _mm256_srli_epi32
#define _MM_SLLI_I _mm256_slli_epi32
#define _MM_ADD_I _mm256_add_epi32
#define _MM_SUB_I _mm256_sub_epi32
#define _MM_CAST_FP_TO_I _mm256_castps_si256
#define _MM_CAST_I_TO_FP _mm256_castsi256_ps
#define _MM_OR  _mm256_or_ps
#define _MM_AND  _mm256_and_ps
#define _MM_ANDNOT  _mm256_andnot_ps
#define _MM_FLOOR  _mm256_floor_ps
#define _MM_LOAD  _mm256_load_ps
#define _MM_LOADU  _mm256_loadu_ps
#define _MM_LOAD3 _mm_load_st3
#define _MM_STORE _mm256_store_ps
#define _MM_STOREU _mm256_storeu_ps
#define _MM_MUL   _mm256_mul_ps
#define _MM_ADD   _mm256_add_ps
#define _MM_SUB   _mm256_sub_ps
#define _MM_DIV   _mm256_div_ps
#define _MM_SQRT  _mm256_sqrt_ps
#define _MM_HADD _mm256_hadd_ps
#define _MM_RHADD _mm256_rhadd_ps // JMCG REAL HADD, totally horizontal
#define _MM_FULL_HADD _mm256_fullhadd_f32
#define _MM_CVT_F _mm_cvtss_f32
#define _MM_CVT_I_TO_FP _mm256_cvtepi32_ps
#define _MM_CVT_FP_TO_I _mm256_cvtps_epi32
#define _MM_SET(A)  _mm256_set1_ps(A)
#define _MM_SETM(A,B,C,D,E,F,G,H)  _mm256_set_ps(A,B,C,D,E,F,G,H)
#define _MM_SET_I(A)  _mm256_set1_epi32(A)
#define _MM_SETM_I(A,B,C,D,E,F,G,H)  _mm256_set_epi32(A,B,C,D,E,F,G,H)
#define _MM_SETR  _mm256_setr_ps
#define _MM_BROADCAST  _mm256_broadcast_ps
#define _MM_MOVEMASK _mm256_movemask_ps
#define _MM_MASK_TRUE 255 // 8 Bits at 1
#define _MM_MAX _mm256_max_ps
#define _MM_MIN _mm256_min_ps
#define _MM_ATAN _mm256_atan_ps
#define _MM_BLENDV _mm256_blendv_ps
#define _MM_COPYSIGN _mm256_copysign_ps // _MM_COPYSIGN(X,Y) takes sign from Y and copies it to X
#define _MM_MALLOC(A,B) _mm_malloc(A,B)
#define _MM_FMA _mm256_fmadd_ps
#define _MM_PRINT_XMM print_xmm
#define _MM_PRINT_XMM_I print_xmm_i


#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (32)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

#ifndef __AVX2__
static inline _MM_TYPE_I _mm256_srli_epi32(_MM_TYPE_I x, uint32_t imm8) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_srli_epi32(_mm256_extractf128_si256(x, 0), imm8);
  __m128i emm02 = _mm_srli_epi32(_mm256_extractf128_si256(x, 1), imm8);
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_slli_epi32(_MM_TYPE_I x, uint32_t imm8) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_slli_epi32(_mm256_extractf128_si256(x, 0), imm8);
  __m128i emm02 = _mm_slli_epi32(_mm256_extractf128_si256(x, 1), imm8);
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_add_epi32(_MM_TYPE_I x, _MM_TYPE_I y) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_add_epi32(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(y, 0));
  __m128i emm02 = _mm_add_epi32(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
static inline _MM_TYPE_I _mm256_sub_epi32(_MM_TYPE_I x, _MM_TYPE_I y) {
  _MM_TYPE_I output;
  __m128i emm01 = _mm_sub_epi32(_mm256_extractf128_si256(x, 0), _mm256_extractf128_si256(y, 0));
  __m128i emm02 = _mm_sub_epi32(_mm256_extractf128_si256(x, 1), _mm256_extractf128_si256(y, 1));
  output = _mm256_insertf128_si256(output, emm01, 0);
  output = _mm256_insertf128_si256(output, emm02, 1);
  return output;
}
#endif

#ifndef __FMA3__
#ifndef _FMAINTRIN_H_INCLUDED
static inline _MM_TYPE _mm256_fmadd_ps(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return _MM_ADD(_MM_MUL(A,B),C);
}
#endif
#endif

__attribute__((aligned (32))) static const int absmask_256[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
#define _mm256_abs_ps(x) _mm256_and_ps((x), *(const __m256*)absmask_256)
__attribute__((aligned (32))) static const int negmask_256[] = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000};
#define _mm256_neg_ps(x) _mm256_xor_ps((x), *(const __m256*)negmask_256)


static inline _MM_TYPE _mm256_copysign_ps(_MM_TYPE x, _MM_TYPE y) {
  return _mm256_or_ps(_MM_ABS(x),_mm256_and_ps(y,*(const __m256*)negmask_256));
}

// For debugging
static inline void print_xmm(_MM_TYPE in, char* s) {
  int i;
  _MM_ALIGN float val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%.16f ", val[i]);
  }
  printf("\n");
}

static inline void print_xmm_i(_MM_TYPE_I in, char* s) {
  int i;
  int32_t val[SIMD_WIDTH];

  _mm256_storeu_si256((__m256i*)&val[0], in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%d ", val[i]);
  }
  printf("\n");
}

static inline _MM_TYPE _mm256_rhadd_ps(_MM_TYPE A, _MM_TYPE B) {

  _MM_TYPE temp = _mm256_hadd_ps(A,B); // B7+B8 B5+B6 A7+A8 A5+A6 B3+B4 B1+B2 A3+A4 A1+A2
  _MM_TYPE temp2 = _mm256_permute2f128_ps( temp , temp , _MM_SHUFFLE2(0,1)); // B3+B4 B1+B2 A3+A4 A1+A2 B7+B8 B5+B6 A7+A8 A5+A6
  _MM_TYPE temp3 = _mm256_shuffle_ps(temp,temp2,_MM_SHUFFLE(1,0,1,0)); // A3+A4 A1+A2 A7+A8 A5+A6 A7+A8 A5+A6 A3+A4 A1+A2
  _MM_TYPE temp4 = _mm256_shuffle_ps(temp2,temp,_MM_SHUFFLE(3,2,3,2)); // B7+B8 B5+B6 B3+B4 B1+B2 B3+B4 B1+B2 B7+B8 B5+B6

  return _mm256_blend_ps(temp3,temp4,0b11110000); // B3+B4 B1+B2 A3+A4 A1+A2
}

static inline __m128 _mm256_fullhadd_f32(_MM_TYPE A, _MM_TYPE B) {
  __m128 hi,lo;
  _MM_TYPE temp = _mm256_hadd_ps(A,B);
  temp = _mm256_hadd_ps(temp,temp);

  hi = _mm256_extractf128_ps(temp,1);
  lo = _mm256_extractf128_ps(temp,0);

  return _mm_add_ps(hi,lo);
}


static inline double _mm256_cvtss_f32(_MM_TYPE A) {
  return (_mm_cvtss_f32(_mm256_extractf128_ps(A,0)));
}

static inline void _mm_load_st3(_MM_TYPE* A, _MM_TYPE* B, _MM_TYPE* C, float* address) {
  _MM_TYPE a_temp, b_temp, c_temp;
  _MM_TYPE aa, bb, cc, ap, bp, cp;

  a_temp = _MM_LOAD(address); // b3,a3,c2,b2,a2,c1,b1,a1
  b_temp = _MM_LOAD(address+8); // a6,c5,b5,a5,c4,b4,a4,c3
  c_temp = _MM_LOAD(address+16); // c8,b8,a8,c7,b7,a7,c6,b6

  // get all a
  aa = _mm256_blend_ps(a_temp,b_temp,0b10110110); // aa = a6,a3,b5,a5,a2,b4,a4,a1
  aa = _mm256_blend_ps(aa,c_temp,0b00100100); // aa = a6,a3,a8,a5,a2,a7,a4,a1

  // get all b
  bb = _mm256_blend_ps(a_temp,b_temp,0b01101101); // bb = b3,c5,b5,b2,c4,b4,b1,c3
  bb = _mm256_blend_ps(bb,c_temp,0b01001001); // bb = b3,b8,b5,b2,b7,b4,b1,b6

  // get all c
  cc = _mm256_blend_ps(a_temp,b_temp,0b11011011); // cc = a6,c5,c2,a5,c4,c1,a4,c3
  cc = _mm256_blend_ps(cc,c_temp,0b10010010); // cc = c8,c5,c2,c7,c4,c1,c6,c3

  // A, Shuffle, permute 2f, blend
  aa = _mm256_shuffle_ps(aa,aa,_MM_SHUFFLE(1,2,3,0)); // aa = a8,a3,a6,a5,a4,a7,a2,a1
  ap = _mm256_permute2f128_ps(aa,aa,0b00000001); // ap = a4,a7,a2,a1,a8,a3,a6,a5
  *A = _mm256_blend_ps(aa,ap,0b01000100); // A = a8,a7,a6,a5,a4,a3,a2,a1

  // B, Shuffle, permute 2f, blend
  bb = _mm256_shuffle_ps(bb,bb,_MM_SHUFFLE(2,3,0,1)); // bb = b8,b3,b2,b5,b4,b7,b6,b1
  bp = _mm256_permute2f128_ps(bb,bb,0b00000001); // bp = b4,b7,b6,b1,b8,b3,b2,b5
  *B = _mm256_blend_ps(bb,bp,0b01100110); // B = b8,b7,b6,b5,b4,b3,b2,b1

  // C, Shuffle, permute 2f, blend
  cc = _mm256_shuffle_ps(cc,cc,_MM_SHUFFLE(3,0,1,2)); // cc = c8,c7,c2,c5,c4,c3,c6,c1
  cp = _mm256_permute2f128_ps(cc,cc,0b00000001); // cp = c4,c3,c6,c1,c8,c7,c2,c5
  *C = _mm256_blend_ps(cc,cp,0b00100010); // C = c8,c7,c6,c5,c4,c3,c2,c1
}


static inline _MM_TYPE_I _custom_mm256_cmpeq_epi32(_MM_TYPE_I A, _MM_TYPE_I B) {
  return (_MM_TYPE_I)_MM_CMPEQ((_MM_TYPE)A,(_MM_TYPE)B);
}


// Algorithm taken from vecmathlib and SLEEF 2.80
static inline _MM_TYPE _mm256_atan_ps(_MM_TYPE A) {
  _MM_TYPE q1 = A;
  _MM_TYPE s = _MM_ABS(A);

  _MM_TYPE q0 = _MM_CMPGT(s,_MM_SET(1.0));
  s = _MM_BLENDV(s,_MM_DIV(_MM_SET(1.0),s),q0); // s = ifthen(q0, rcp(s), s);

  _MM_TYPE t = _MM_MUL(s,s); //  realvec_t t = s * s;
  _MM_TYPE u = _MM_SET(0.00282363896258175373077393f);

  u = _MM_FMA(u,t,_MM_SET(-0.0159569028764963150024414f));
  u = _MM_FMA(u,t,_MM_SET(0.0425049886107444763183594f));
  u = _MM_FMA(u,t,_MM_SET(-0.0748900920152664184570312f));
  u = _MM_FMA(u,t,_MM_SET(0.106347933411598205566406f));
  u = _MM_FMA(u,t,_MM_SET(-0.142027363181114196777344f));
  u = _MM_FMA(u,t,_MM_SET(0.199926957488059997558594f));
  u = _MM_FMA(u,t,_MM_SET(-0.333331018686294555664062f));

  t = _MM_ADD(s,_MM_MUL(s,_MM_MUL(t,u))); //  t = s + s * (t * u);

  t = _MM_BLENDV(t,_MM_SUB(_MM_SET(M_PI_2),t),q0); // t = ifthen(q0, RV(M_PI_2) - t, t);
  t = _MM_COPYSIGN(t, q1);

  return t;
}


#endif // PARSEC_USE_AVX

#ifdef PARSEC_USE_NEON
#include <arm_neon.h> // ALL NEON instructions

#define _MM_ALIGNMENT 16
#define _MM_MANTISSA_BITS 23
#define _MM_MANTISSA_MASK 0x007fffff
#define _MM_EXP_MASK 0x7f800000
#define _MM_EXP_BIAS 0x7f
#define _MM_MINNORMPOS (1 << _MM_MANTISSA_BITS)
#define _MM_TYPE float32x4_t
#define _MM_TYPE_I int32x4_t
#define _MM_SCALAR_TYPE float
#define SIMD_WIDTH 4
#define _MM_SETZERO vsetzeroq_f32 // not available
#define _MM_SETZERO_I vsetzeroq_s32 // not available
#define _MM_ABS vabsq_f32
#define _MM_NEG vnegq_f32
#define _MM_LOG simd_log // not available
#define _MM_EXP simd_exp // not available
#define _MM_CMPGT vcgtq_f32
#define _MM_CMPLT vcltq_f32
#define _MM_CMPLE vcleq_f32
#define _MM_CMPEQ vceqq_f32
#define _MM_CMPEQ_SIG vceqq_s32
#define _MM_SRLI_I vshrq_n_s32
#define _MM_SLLI_I vshlq_n_s32
#define _MM_ADD_I vaddq_s32
#define _MM_SUB_I vsubq_s32
#define _MM_CAST_FP_TO_I vreinterpretq_s32_f32
#define _MM_CAST_I_TO_FP vreinterpretq_f32_s32
#define _MM_OR  vorrq_f32 // not available
#define _MM_AND  vandq_f32 // not available
#define _MM_ANDNOT  vbicq_f32 // not available
#define _MM_FLOOR  vfloorq_f32 // not available
#define _MM_LOAD  vld1q_f32
#define _MM_LOADU vld1q_f32 // Not completely sure about how to deal with this
#define _MM_LOAD3 custom_vld3q_f32 // Although ARM supports stride loads, the format of the intrinsics is quite weird
#define _MM_STORE vst1q_f32
#define _MM_STOREU vst1q_f32 // Not completely sure about how to deal with this
#define _MM_MUL vmulq_f32
#define _MM_ADD vaddq_f32
#define _MM_SUB vsubq_f32
#define _MM_DIV vdivq_f32 // not available
#define _MM_SQRT vsqrtq_f32 // not available
#define _MM_HADD vhoriaddq_f32 // not available
#define _MM_RHADD vhoriaddq_f32 // not available JMCG REAL HADD, totally horizontal
#define _MM_FULL_HADD vfhoriaddq_f32 // not available
#define _MM_CVT_F vcvtq32_f32 // not available
#define _MM_CVT_I_TO_FP vcvtq_f32_s32 // not available
#define _MM_CVT_FP_TO_I vcvtq_s32_f32 // not available
#define _MM_SET(A) vdupq_n_f32(A)
#define _MM_SETM(A,B,C,D) vsetmq_f32(A,B,C,D) // not available
#define _MM_SET_I(A) vdupq_n_s32(A)
#define _MM_SETM_I(A,B,C,D) vsetmq_s32(A,B,C,D) // not available
#define _MM_SETR vsetrq_f32 // not available
#define _MM_MOVEMASK vmovemaskq_f32
#define _MM_MASK_TRUE 15 // 8 Bits at 1
#define _MM_MAX vmaxq_f32
#define _MM_MIN vminq_f32
#define _MM_BLENDV _vbslq_f32
#define _MM_COPYSIGN vcopysignq_f32 // _MM_COPYSIGN(X,Y) takes sign from Y and copies it to X
#define _MM_FMA _vmlaq_f32
#define _MM_PRINT_XMM print_xmm
#define _MM_PRINT_XMM_I print_xmm_i


#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (16)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

// For debugging
static inline void print_xmm(_MM_TYPE in, char* s) {
  int i;
  _MM_ALIGN float val[SIMD_WIDTH];

  _MM_STORE(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%.16f ", val[i]);
  }
  printf("\n");
}

static inline void print_xmm_i(_MM_TYPE_I in, char* s) {
  int i;
  int32_t val[SIMD_WIDTH];

  vst1q_s32(val, in);
  printf("%s: ", s);
  for (i=0; i<SIMD_WIDTH; i++) {
    printf("%d ", val[i]);
  }
  printf("\n");
}



static inline _MM_TYPE _vbslq_f32(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return vbslq_f32(vcvtq_u32_f32(C),A,B);
}

static inline _MM_TYPE _vmlaq_f32(_MM_TYPE A, _MM_TYPE B, _MM_TYPE C) {
  return vmlaq_f32(C,A,B);
}

static inline _MM_TYPE vsetzeroq_f32() {
  return vdupq_n_f32(0);
}

static inline _MM_TYPE vsetzeroq_s32() {
  return vdupq_n_s32(0);
}

static inline _MM_TYPE vorrq_f32(_MM_TYPE A, _MM_TYPE B) {
  return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(A), vreinterpretq_u32_f32(B)));
}

static inline _MM_TYPE vandq_f32(_MM_TYPE A, _MM_TYPE B) {
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(A), vreinterpretq_u32_f32(B)));
}

static inline _MM_TYPE vbicq_f32(_MM_TYPE A, _MM_TYPE B) {
  return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(B), vreinterpretq_u32_f32(A)));
}

static inline _MM_TYPE vfloorq_f32(_MM_TYPE A) {
  return vreinterpretq_f32_u32((vreinterpretq_u32_f32(A)));
}

static inline _MM_TYPE vdivq_f32(_MM_TYPE A, _MM_TYPE B) {
  _MM_TYPE _tmp1f = vrecpeq_f32(B);
  _tmp1f = vmulq_f32(vrecpsq_f32(B, _tmp1f), _tmp1f);
  _tmp1f = vmulq_f32(vrecpsq_f32(B, _tmp1f), _tmp1f);
  _tmp1f = vmulq_f32(vrecpsq_f32(B, _tmp1f), _tmp1f);
  _tmp1f = vmulq_f32(vrecpsq_f32(B, _tmp1f), _tmp1f);
  _tmp1f = vmulq_f32(A,_tmp1f);
  return _tmp1f;
}

static inline _MM_TYPE vsqrtq_f32(_MM_TYPE A) {
  // Inverse sqrt
  _MM_TYPE _tmp1f = vrsqrteq_f32(A);
  _tmp1f = vmulq_f32(_tmp1f, vrsqrtsq_f32(vmulq_f32(_tmp1f,_tmp1f), A));
  _tmp1f = vmulq_f32(_tmp1f, vrsqrtsq_f32(vmulq_f32(_tmp1f,_tmp1f), A));
  _tmp1f = vmulq_f32(_tmp1f, vrsqrtsq_f32(vmulq_f32(_tmp1f,_tmp1f), A));
  _tmp1f = vmulq_f32(_tmp1f, vrsqrtsq_f32(vmulq_f32(_tmp1f,_tmp1f), A));
  // Invert
  _MM_TYPE sqrt = vrecpeq_f32(_tmp1f);
  sqrt = vmulq_f32(sqrt, vrecpsq_f32(sqrt, _tmp1f));
  sqrt = vmulq_f32(sqrt, vrecpsq_f32(sqrt, _tmp1f));
  sqrt = vmulq_f32(sqrt, vrecpsq_f32(sqrt, _tmp1f));
  sqrt = vmulq_f32(sqrt, vrecpsq_f32(sqrt, _tmp1f));

  return sqrt;
}


static inline _MM_TYPE vhoriaddq_f32(_MM_TYPE A, _MM_TYPE B) {
  // Unluckily I cannot find a better way to do this rather than putting together
  // the high and low parts, that is bad in performance as this instrucion is usually
  // followed by vcvtqf32_f32
  float32x2_t A1 = vpadd_f32(vget_low_f32(A),vget_high_f32(A));
  float32x2_t B1 = vpadd_f32(vget_low_f32(B),vget_high_f32(B));
  return vcombine_f32(A1,B1);
}

static inline _MM_TYPE vfhoriaddq_f32(_MM_TYPE A, _MM_TYPE B) {
  // Unluckily I cannot find a better way to do this rather than putting together
  // the high and low parts, that is bad in performance as this instrucion is usually
  // followed by vcvtqf32_f32
  float32x2_t A1 = vpadd_f32(vget_low_f32(A),vget_high_f32(A));
  float32x2_t B1 = vpadd_f32(vget_low_f32(B),vget_high_f32(B));
  float32x2_t AB = vpadd_f32(A1,B1);
  return vcombine_f32(AB,AB);
}


static inline float vcvtq32_f32(_MM_TYPE A) {
  return vgetq_lane_f32(A, 0);
}

static inline _MM_TYPE vsetmq_f32(float A, float B, float C, float D) {
  _MM_TYPE temp = {D, C, B, A};
  return temp;
}

static inline _MM_TYPE vsetrq_f32(float A, float B, float C, float D) {
  _MM_TYPE temp = {A, B, C, D};
  return temp;
}

static inline void custom_vld3q_f32(_MM_TYPE* A, _MM_TYPE* B, _MM_TYPE* C, float* address) {
  float32x4x3_t temp;

  temp = vld3q_f32((const float32_t *)address);
  *A = temp.val[0];
  *B = temp.val[1];
  *C = temp.val[2];
}

// This is a 1 cycle latency instruction in Intel architectures, for ARM it takes 5 instructions to simulate
// we can expect bad performance
static inline int vmovemaskq_f32( _MM_TYPE in ) {
  // Get signs
  uint32x4_t signmask = {0x80000000, 0x80000000, 0x80000000, 0x80000000};
  uint32x4_t temp = vtstq_u32 (vreinterpretq_u32_f32(in), signmask);

  // Create addition mask
  const uint32x4_t qMask = { 1, 2, 4, 8 };
  const uint32x4_t qAnded = vandq_u32( temp, qMask ); // Get int values based on sign

  const uint32x2_t dHigh = vget_high_u32( qAnded );
  const uint32x2_t dLow = vget_low_u32( qAnded );

  const uint32x2_t dOred = vorr_u32( dHigh, dLow ); // combine results
  const uint32x2_t dMask = vpadd_u32( dOred, dOred ); // Horizotal add
  return vget_lane_u32( dMask, 0 );
}

__attribute__((aligned (16))) static const int negmask[] = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

static inline _MM_TYPE vcopysignq_f32(_MM_TYPE x, _MM_TYPE y) {
  return (_MM_TYPE)vorrq_s32((_MM_TYPE_I)_MM_ABS(x),vandq_s32((_MM_TYPE_I)y,*(const int32x4_t*)negmask));
}

#endif // PARSEC_USE_NEON

#endif // DFTYPE

#if defined(SIMD_WIDTH) && !defined(PARSEC_USE_NEON)
#include <mm_malloc.h>
#else
// Code taken from gcc 4.8 mm_malloc.h
#ifndef _MM_MALLOC_H_INCLUDED
#define _MM_MALLOC_H_INCLUDED

#include <stdlib.h>

/* We can't depend on <stdlib.h> since the prototype of posix_memalign
   may not be visible.  */
#ifndef __cplusplus
extern int posix_memalign (void **, size_t, size_t);
#else
extern "C" int posix_memalign (void **, size_t, size_t) throw ();
#endif

static __inline void *
_mm_malloc (size_t size, size_t alignment)
{
  void *ptr;
  if (alignment == 1)
    return malloc (size);
  if (alignment == 2 || (sizeof (void *) == 8 && alignment == 4))
    alignment = sizeof (void *);
  if (posix_memalign (&ptr, alignment, size) == 0)
    return ptr;
  else
    return NULL;
}

static __inline void
_mm_free (void * ptr)
{
  free (ptr);
}

#endif /* _MM_MALLOC_H_INCLUDED */
#endif // defined SIMD_WIDTH

#ifndef _MM_ALIGNMENT
#define _MM_ALIGNMENT 1
#endif

#ifdef SIMD_WIDTH
#include "simd_mathfun.h"
#endif

#endif //__SIMD_DEFINES__
