/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
*/

/* Copyright (C) 2007  Julien Pommier

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

  (this is the zlib license)
*/

// AVX additions made by Hallgeir Lien (hallgeir.lien@gmail.com)
// DP and ABS-NEG SIMD Version by Juan M. Cebrian, NTNU - 2013.
//

#ifndef __AVX_MATHFUN__
#define __AVX_MATHFUN__

#include <immintrin.h>
#include "simd_defines.h"

/* yes I know, the top of this file is quite ugly */

#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
# define ALIGN32_BEG
# define ALIGN32_END __attribute__((aligned(32)))
#endif



/* __m128 is ugly to write */
#define USE_SSE2

#ifdef USE_SSE2
# include <emmintrin.h>
typedef __m128i v4si; // vector of 4 int (sse2)
#endif

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN32_BEG float _256ps_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN32_BEG Type _256ps_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#define _PD_CONST(Name, Val) \
  static const ALIGN32_BEG double _256pd_##Name[4] ALIGN32_END = { Val, Val, Val, Val }

#define _PD_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN32_BEG Type _256pd_##Name[4] ALIGN32_END = { Val, Val, Val, Val }

#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val };\
  static const ALIGN32_BEG int _256pi32_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

#define DECLARE_LONG_CONST(Name, Val1, Val2) \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END ALIGN16_END = { Val1, Val2, Val1, Val2 }; \
  static const ALIGN32_BEG int _256pi32_##Name[8] ALIGN32_END ALIGN32_END = { Val1, Val2, Val1, Val2, Val1, Val2, Val1, Val2 };

_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

/*
    BEGIN MODIFICATIONS
    AVX additions made by Hallgeir Lien (hallgeir.lien@gmail.com)
*/

typedef __m256 v8sf;
typedef __m256i v8si;

/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/

static inline v8sf log256_ps(v8sf x) {
  v8si emm0;
  v8sf one = *(v8sf*)_256ps_1;

  v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
  x = _mm256_max_ps(x, *(v8sf*)_256ps_min_norm_pos);  /* cut off denormalized stuff */
  //256 bit shift is not implemented yet; do two 128 bit shifts
  {
//      _mm256_zeroupper();
      v4si emm01 = _mm_srli_epi32(_mm_castps_si128(_mm256_extractf128_ps(x, 0)), 23);
      v4si emm02 = _mm_srli_epi32(_mm_castps_si128(_mm256_extractf128_ps(x, 1)), 23);
      //256 bit arithmetic not implemented... do it separately
      emm01 = _mm_sub_epi32(emm01, *(v4si*)_pi32_0x7f);
      emm02 = _mm_sub_epi32(emm02, *(v4si*)_pi32_0x7f);

      emm0 = _mm256_insertf128_si256(emm0, emm01, 0);
      emm0 = _mm256_insertf128_si256(emm0, emm02, 1);
  }

  /* keep only the fractional part */
  x = _mm256_and_ps(x, *(v8sf*)_256ps_inv_mant_mask);
  x = _mm256_or_ps(x, *(v8sf*)_256ps_0p5);
  v8sf e = _mm256_cvtepi32_ps(emm0);

  e = _mm256_add_ps(e, one);
  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  v8sf mask = _mm256_cmplt_ps(x, *(v8sf*)_256ps_cephes_SQRTHF);
  v8sf tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);


  v8sf z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_256ps_cephes_log_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p5);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p6);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p7);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_log_p8);
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);


  tmp = _mm256_mul_ps(e, *(v8sf*)_256ps_cephes_log_q1);
  y = _mm256_add_ps(y, tmp);


  tmp = _mm256_mul_ps(z, *(v8sf*)_256ps_0p5);
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, *(v8sf*)_256ps_cephes_log_q2);
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

_PS_CONST(exp_hi,       88.3762626647949f);
_PS_CONST(exp_lo,       -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);


static inline v8sf exp256_ps(v8sf x) {
  v8sf tmp = _mm256_setzero_ps(), fx;
  v8si emm0 = _mm256_setzero_si256();
  v8sf one = *(v8sf*)_256ps_1;

  x = _mm256_min_ps(x, *(v8sf*)_256ps_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_256ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *(v8sf*)_256ps_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_256ps_0p5);

  /* how to perform a floorf with SSE: just below */
  emm0 = _mm256_cvttps_epi32(fx);
  tmp  = _mm256_cvtepi32_ps(emm0);

  /* if greater, substract 1 */
  v8sf mask = _mm256_cmpgt_ps(tmp, fx);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, *(v8sf*)_256ps_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_256ps_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_256ps_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_256ps_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  emm0 = _mm256_cvttps_epi32(fx);
  {
      v4si emm01 = _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256(emm0, 0), *(v4si*)_pi32_0x7f), 23),
           emm02 = _mm_slli_epi32(_mm_add_epi32(_mm256_extractf128_si256(emm0, 1), *(v4si*)_pi32_0x7f), 23);

      emm0 = _mm256_insertf128_si256(emm0, emm01, 0);
      emm0 = _mm256_insertf128_si256(emm0, emm02, 1);
  }
  v8sf pow2n = _mm256_castsi256_ps(emm0);
  y = _mm256_mul_ps(y, pow2n);

  return y;
}
//#endif
/*
    END MODIFICATIONS
*/


/* JMCG */

#define MANTISSA_BITS_DOUBLE 52

_PD_CONST(LOG_P0, 7.0376836292E-2);
_PD_CONST(LOG_P1, - 1.1514610310E-1);
_PD_CONST(LOG_P2, 1.1676998740E-1);
_PD_CONST(LOG_P3, - 1.2420140846E-1);
_PD_CONST(LOG_P4, + 1.4249322787E-1);
_PD_CONST(LOG_P5, - 1.6668057665E-1);
_PD_CONST(LOG_P6, + 2.0000714765E-1);
_PD_CONST(LOG_P7, - 2.4999993993E-1);
_PD_CONST(LOG_P8, + 3.3333331174E-1);
_PD_CONST(LOG_Q1, -2.12194440e-4);
_PD_CONST(LOG_Q2, 0.693359375);
_PD_CONST(SQRTHALF, 0.707106781186547524);

_PD_CONST(ONE, 1.0);
_PD_CONST(NEGONE, -1.0);
_PD_CONST(TWO, 2.0);
_PD_CONST(HALF, 0.5);

DECLARE_LONG_CONST(SIGN_MASK_DOUBLE, 0x00000000, 0x80000000);
#define SIGN_MASK_DOUBLE *(__m128i*) _pi32_SIGN_MASK_DOUBLE

DECLARE_LONG_CONST(INV_SIGN_MASK_DOUBLE, 0xffffffff, 0x7fffffff);
#define INV_SIGN_MASK_DOUBLE *(__m128i*) _pi32_INV_SIGN_MASK_DOUBLE

DECLARE_LONG_CONST(MINNORMPOS_DOUBLE, 0x00000000, 1 << (MANTISSA_BITS_DOUBLE - 32));
#define MINNORMPOS_DOUBLE *(__m128i*) _pi32_MINNORMPOS_DOUBLE

DECLARE_LONG_CONST(MANTISSAMASK_DOUBLE, 0x00000000, (0xffffffff << (MANTISSA_BITS_DOUBLE - 32)) & 0x7fffffff);
#define MANTISSAMASK_DOUBLE *(__m128i*) _pi32_MANTISSAMASK_DOUBLE

DECLARE_LONG_CONST(INVMANTISSAMASK_DOUBLE, 0xffffffff, ~((0xffffffff << (MANTISSA_BITS_DOUBLE - 32)) & 0x7fffffff));
#define INVMANTISSAMASK_DOUBLE *(__m128i*) _pi32_INVMANTISSAMASK_DOUBLE

// exponent bias in double precision (1023)
DECLARE_LONG_CONST(EXPBIAS_DOUBLE, 0x000003ff, 0x00000000);
#define EXPBIAS_DOUBLE *(__m128i*) _pi32_EXPBIAS_DOUBLE

#define cvtepi64_pd(x) _mm_cvtepi32_pd ((__m128i) _mm_shuffle_ps ((__m128) (x), (__m128) (x), _MM_SHUFFLE (3, 3, 2, 0)))

#define SIGN_MASK_DOUBLE_256 *(__m256i*) _256pi32_SIGN_MASK_DOUBLE
#define INV_SIGN_MASK_DOUBLE_256 *(__m256i*) _256pi32_INV_SIGN_MASK_DOUBLE
#define MINNORMPOS_DOUBLE_256 *(__m256i*) _256pi32_MINNORMPOS_DOUBLE
#define MANTISSAMASK_DOUBLE_256 *(__m256i*) _256pi32_MANTISSAMASK_DOUBLE
#define INVMANTISSAMASK_DOUBLE_256 *(__m256i*) _256pi32_INVMANTISSAMASK_DOUBLE
// exponent bias in double precision (1023)
#define EXPBIAS_DOUBLE_256 *(__m256i*) _256pi32_EXPBIAS_DOUBLE

static inline __m256d log256_pd (__m256d x)
{
  __m256i emm0;
  __m256d e;
  //  __m256d invalid_mask = _mm256_cmple_pd (x, _mm256_setzero_pd ());
  __m256d invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_LE_OS);

  // cut off denormalized stuff
  x = _mm256_max_pd (x, (__m256d) MINNORMPOS_DOUBLE_256);

  emm0 = _mm256_castpd_si256 (x);


  //  emm0 = _mm256_srli_epi64 (emm0, MANTISSA_BITS_DOUBLE_256);
  //256 bit shift is not implemented yet; do two 128 bit shifts
  {
    //      _mm256_zeroupper();
    v4si emm01 = _mm_srli_epi64(_mm_castpd_si128(_mm256_extractf128_pd(x, 0)), MANTISSA_BITS_DOUBLE);
    v4si emm02 = _mm_srli_epi64(_mm_castpd_si128(_mm256_extractf128_pd(x, 1)), MANTISSA_BITS_DOUBLE);
    //256 bit arithmetic not implemented... do it separately

    //  emm0 = _mm256_sub_epi64 (emm0, EXPBIAS_DOUBLE_256);

    emm01 = _mm_sub_epi64 (emm01, EXPBIAS_DOUBLE);
    emm02 = _mm_sub_epi64 (emm02, EXPBIAS_DOUBLE);

    __m128d e1 = cvtepi64_pd (emm01);
    __m128d e2 = cvtepi64_pd (emm02);

    e = _mm256_insertf128_pd(e, e1, 0);
    e = _mm256_insertf128_pd(e, e2, 1);

  }


  // keep only the fractional part
  x = _mm256_and_pd (x, (__m256d) INVMANTISSAMASK_DOUBLE_256);
  x = _mm256_or_pd (x, *(__m256d*)_256pd_HALF);

  e = _mm256_add_pd (e, *(__m256d*)_256pd_ONE);

  //  __m256d mask = _mm256_cmplt_pd (x, _256pd_SQRTHALF);
  __m256d mask = _mm256_cmp_pd (x, *(__m256d*)_256pd_SQRTHALF, _CMP_LT_OS);

  __m256d tmp = _mm256_and_pd (x, mask);
  x = _mm256_sub_pd (x, *(__m256d*)_256pd_ONE);
  e = _mm256_sub_pd (e, _mm256_and_pd (*(__m256d*)_256pd_ONE, mask));
  x = _mm256_add_pd (x, tmp);

  __m256d z = _mm256_mul_pd (x, x);

  __m256d y = *(__m256d*)_256pd_LOG_P0;
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P1);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P2);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P3);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P4);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P5);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P6);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P7);
  y = _mm256_mul_pd (y, x);
  y = _mm256_add_pd (y, *(__m256d*)_256pd_LOG_P8);
  y = _mm256_mul_pd (y, x);

  y = _mm256_mul_pd (y, z);

  tmp = _mm256_mul_pd (e, *(__m256d*)_256pd_LOG_Q1);
  y = _mm256_add_pd (y, tmp);

  tmp = _mm256_mul_pd (z, *(__m256d*)_256pd_HALF);
  y = _mm256_sub_pd (y, tmp);

  tmp = _mm256_mul_pd (e, *(__m256d*)_256pd_LOG_Q2);
  x = _mm256_add_pd (x, y);
  x = _mm256_add_pd (x, tmp);

  // set negative args to be NaN
  x = _mm256_or_pd (x, invalid_mask);

  return x;
}


/*
__attribute__((aligned (32))) static const int absmask_256[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
__attribute__((aligned (32))) static const int absmask_double_256[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};
#define _mm256_abs_ps(x) _mm256_and_ps((x), *(const __m256*)absmask_256)
#define _mm256_abs_pd(x) _mm256_and_pd((x), *(const __m256d*)absmask_double_256)


__attribute__((aligned (32))) static const int negmask_256[] = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000};
__attribute__((aligned (32))) static const int negmask_double_256[] = { 0xffffffff, 0x80000000, 0xffffffff, 0x80000000, 0xffffffff, 0x80000000, 0xffffffff, 0x80000000};
#define _mm256_neg_ps(x) _mm256_xor_ps((x), *(const __m256*)negmask_256)
#define _mm256_neg_pd(x) _mm256_xor_pd((x), *(const __m256d*)negmask_double_256)
*/


/* END JMCG */
#endif // __SSE_MATHFUN__
