#ifndef _SIMD_H_
#define _SIMD_H_

#include <xmmintrin.h>
#include <assert.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <nmmintrin.h>

/*
	various things that are useful/necessary for SIMD execution in general.
 */

// attempted 256bit SIMD with AVX, but need AVX2 for integer 256bit ops
// AVX2 is released in 2013, but none of our machines support it yet
// comment out to avoid breaking things - yjo 8/10/13
//#ifdef AVX
//#include <immintrin.h>
//const int SIMD_WIDTH  = 8;
//const int FULL_MASK = 0xff;
//#define _mm256_load1_ps	_mm256_broadcast_ss
//#else
//const int SIMD_WIDTH  = 4;
//const int FULL_MASK = 0xf;
//#endif

/* CDF START */
#ifdef PARSEC_USE_SSE
	const int LOCAL_SIMD_WIDTH  = 16;
	const int FULL_MASK = 0xffff;
#endif
#ifdef PARSEC_USE_AVX
	const int LOCAL_SIMD_WIDTH  = 32;
	const int FULL_MASK = 0xffffffff;
#endif
/* CDF END */

//BLEND: res[i] = (mask[i] == 0) ? a[i] : b[i]
//#ifdef SSE41
//#include <smmintrin.h>
//#define BLENDV_PS(a, b, mask) _mm_blendv_ps(a, b, mask)
//#else
//#define BLENDV_PS(a, b, mask) _mm_or_ps(_mm_andnot_ps(mask, a),_mm_and_ps(mask, b))
//#endif
//
//#define ALL_SET _mm_cmpeq_ps(_mm_setzero_ps(), _mm_setzero_ps())
//#define NOT_PS(a) _mm_xor_ps(a, ALL_SET)
//
//typedef union _simdunion {
//	__m128 f;
//	__m128i i;
//} __m128both;
//
//int get_valid_mask(int _si, int size);
//
#endif
