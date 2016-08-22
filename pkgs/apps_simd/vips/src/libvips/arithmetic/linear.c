/* im_lintra.c -- linear transform
 *
 * Copyright: 1990, N. Dessipris, based on im_powtra()
 * Author: Nicos Dessipris
 * Written on: 02/05/1990
 * Modified on:
 * 23/4/93 JC
 *	- adapted to work with partial images
 * 1/7/93 JC
 *	- adapted for partial v2
 * 7/10/94 JC
 *	- new IM_NEW()
 *	- more typedefs
 * 9/2/95 JC
 *	- adapted for im_wrap...
 *	- operations on complex images now just transform the real channel
 * 29/9/95 JC
 *	- complex was broken
 * 15/4/97 JC
 *	- return(0) missing from generate, arrgh!
 * 1/7/98 JC
 *	- im_lintra_vec added
 * 3/8/02 JC
 *	- fall back to im_copy() for a == 1, b == 0
 * 10/10/02 JC
 *	- auug, failing to multiply imag for complex! (thanks matt)
 * 10/12/02 JC
 *	- removed im_copy() fallback ... meant that output format could change
 *	  with value :-( very confusing
 * 30/6/04
 *	- added 1 band image * n band vector case
 * 8/12/06
 * 	- add liboil support
 * 9/9/09
 * 	- gtkdoc comment, minor reformat
 * 31/7/10
 * 	- remove liboil
 * 31/10/11
 * 	- rework as a class
 * 	- removed the 1-ary constant path, no faster
 * 30/11/13
 * 	- 1ary is back, faster with gcc 4.8
 * 3/12/13
 * 	- try an ORC path with the band loop unrolled
 * 14/1/14
 * 	- add uchar output option
 */

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

/*

    Copyright (C) 1991-2005 The National Gallery

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301  USA

 */

/*

    These files are distributed with VIPS - http://www.vips.ecs.soton.ac.uk

 */

/*
#define DEBUG
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/
#include <vips/intl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vips/vips.h>

#include "unary.h"

/* JMCG BEGIN */
//#define DFTYPE
#include "simd_defines.h"
#include <assert.h>

//#define DEBUG_SIMD
#ifdef DEBUG_SIMD
#include <math.h>
float diff_l = 0.0f;
float max_diff_l = 0.0f;
#endif
/* JMCG END */

typedef struct _VipsLinear {
	VipsUnary parent_instance;

	/* Our constants: multiply by a, add b.
	 */
	VipsArea *a;
	VipsArea *b;

	/* uchar output.
	 */
	gboolean uchar;

	/* Our constants expanded to match arith->ready in size.
	 */
	int n;
	double *a_ready;
	double *b_ready;

} VipsLinear;

typedef VipsUnaryClass VipsLinearClass;

G_DEFINE_TYPE( VipsLinear, vips_linear, VIPS_TYPE_UNARY );

static int
vips_linear_build( VipsObject *object )
{
	VipsObjectClass *class = VIPS_OBJECT_GET_CLASS( object );
	VipsArithmetic *arithmetic = VIPS_ARITHMETIC( object );
	VipsUnary *unary = (VipsUnary *) object;
	VipsLinear *linear = (VipsLinear *) object;

	int i;

	/* If we have a three-element vector we need to bandup the image to
	 * match.
	 */
	linear->n = 1;
	if( linear->a )
		linear->n = VIPS_MAX( linear->n, linear->a->n );
	if( linear->b )
		linear->n = VIPS_MAX( linear->n, linear->b->n );
	if( unary->in ) {
		int bands;

		vips_image_decode_predict( unary->in, &bands, NULL );
		linear->n = VIPS_MAX( linear->n, bands );
	}
	arithmetic->base_bands = linear->n;

	if( unary->in &&
		linear->a &&
		linear->b ) {
		if( vips_check_vector( class->nickname,
			linear->a->n, unary->in ) ||
			vips_check_vector( class->nickname,
				linear->b->n, unary->in ) )
		return( -1 );
	}

	/* Make up-banded versions of our constants.
	 */
	linear->a_ready = VIPS_ARRAY( linear, linear->n, double );
	linear->b_ready = VIPS_ARRAY( linear, linear->n, double );

	for( i = 0; i < linear->n; i++ ) {
		if( linear->a ) {
			double *ary = (double *) linear->a->data;
			int j = VIPS_MIN( i, linear->a->n - 1 );

			linear->a_ready[i] = ary[j];
		}

		if( linear->b ) {
			double *ary = (double *) linear->b->data;
			int j = VIPS_MIN( i, linear->b->n - 1 );

			linear->b_ready[i] = ary[j];
		}
	}

	if( linear->uchar )
		arithmetic->format = VIPS_FORMAT_UCHAR;

	if( VIPS_OBJECT_CLASS( vips_linear_parent_class )->build( object ) )
		return( -1 );

	return( 0 );
}

/* JMCG BEGIN */
#ifdef SIMD_WIDTH
// JMCG Common for NEON, SSE and AVX

// Validation function for loopn_$SIMD_uchar
static void inline validate_loopn_uchar(double *a, double *b, int width, unsigned char *p, float *q) {
  int i, x;
  for( i = 0, x = 0; x < width; x++ ) {
    q[i] = a[0] * (unsigned char) p[i] + b[0];
    q[i+1] = a[1] * (unsigned char) p[i+1] + b[1];
    q[i+2] = a[2] * (unsigned char) p[i+2] + b[2];
    i += 3;
  }
}

// Validation function for loopn_$SIMD_float
static void inline validate_loopn_float(double *a, double *b, int width, float *p, float *q) {
  int i, x;
  for( i = 0, x = 0; x < width; x++ ) {
    q[i] = a[0] * (float) p[i] + b[0];
    q[i+1] = a[1] * (float) p[i+1] + b[1];
    q[i+2] = a[2] * (float) p[i+2] + b[2];
    i += 3;
  }
}

#endif

#ifdef PARSEC_USE_AVX

// JMCG Vectorization - LOOPN for UCHAR (IN = unsigned char, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_avx_uchar(double *a, double *b, int width, PEL *in, PEL *out) {
  unsigned char *p = (unsigned char *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m128d a_01, a_20, a_12, b_01, b_20, b_12;
  __m256d a_0210, a_1021, a_2102, b_0210, b_1021, b_2102;

  __m256d p_1,p_2,p_3;
  __m128i temp_l;
  int i, x;

  // JMCG
  // Since we are only vectorizing for NB = 3 and can only fit 2 doubles per register on SSE
  // we unrolled 12 times to minimized loads of uchars, but in this final version,
  // as we use *(__m128i *) (&p[i]) is not really necessary, we could unroll 6 times only as we do for floats


  // Since we are only vectorizing for NB = 3 and can only fit 2 doubles per register on SSE, we unroll 6 times
  a_01 = _mm_loadu_pd(&a[0]); // GR
  a_12 = _mm_loadu_pd(&a[1]); // BG
  a_20 = _mm_loadh_pd(_mm_load_sd(&a[2]),
                      &a[0]); // RB

  a_0210 = _mm256_insertf128_pd(a_0210,a_01,0);
  a_0210 = _mm256_insertf128_pd(a_0210,a_20,1); // RBGR

  a_1021 = _mm256_insertf128_pd(a_1021,a_12,0);
  a_1021 = _mm256_insertf128_pd(a_1021,a_01,1); // GRBG

  a_2102 = _mm256_insertf128_pd(a_2102,a_20,0);
  a_2102 = _mm256_insertf128_pd(a_2102,a_12,1); // BGRB

  b_01 = _mm_loadu_pd(&b[0]); // GR
  b_12 = _mm_loadu_pd(&b[1]); // BG
  b_20 = _mm_loadh_pd(_mm_load_sd(&b[2]),
                      &b[0]); // RG

  b_0210 = _mm256_insertf128_pd(b_0210,b_01,0);
  b_0210 = _mm256_insertf128_pd(b_0210,b_20,1); // RBGR

  b_1021 = _mm256_insertf128_pd(b_1021,b_12,0);
  b_1021 = _mm256_insertf128_pd(b_1021,b_01,1); // GRBG

  b_2102 = _mm256_insertf128_pd(b_2102,b_20,0);
  b_2102 = _mm256_insertf128_pd(b_2102,b_12,1); // BGRB

  i = 0;
  x = 0;

  while ((x + 4) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i]));
    p_1 = _mm256_cvtepi32_pd(temp_l); // R2B1G1R1

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i+4]));
    p_2 = _mm256_cvtepi32_pd(temp_l); // G3R3B2G2

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i+8]));
    p_3 = _mm256_cvtepi32_pd(temp_l); // B4G4R4B3

    p_1 = _mm256_add_pd(_mm256_mul_pd(p_1,a_0210),b_0210); // (R2B1G1R1 * A0A2A1A0) + B0B2B1B0
    p_2 = _mm256_add_pd(_mm256_mul_pd(p_2,a_1021),b_1021); // (G3R3B2G2 * A1A0A2A1) + B1B0B2B1
    p_3 = _mm256_add_pd(_mm256_mul_pd(p_3,a_2102),b_2102); // (B4G4R4B3 * A2A1A0A2) + B2B1B0B2

    _mm_storeu_ps(&q[i],_mm256_cvtpd_ps(p_1));
    _mm_storeu_ps(&q[i+4],_mm256_cvtpd_ps(p_2));
    _mm_storeu_ps(&q[i+8],_mm256_cvtpd_ps(p_3));

    i += 12;
    x += 4;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (unsigned char) p[i]) + b[0];
    q[i+1] = (a[1] * (unsigned char) p[i+1]) + b[1];
    q[i+2] = (a[2] * (unsigned char) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_uchar(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Uchar Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff UChar = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

// JMCG Vectorization - LOOPN for FLOAT (IN = float, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_avx_float(double *a, double *b, int width, PEL *in, PEL *out) {
  float *p = (float *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m128d a_01, a_20, a_12, b_01, b_20, b_12;
  __m256d a_0210, a_1021, a_2102, b_0210, b_1021, b_2102;

  __m256d p_1,p_2,p_3;
  __m128 temp;
  int i, x;

  // Since we are only vectorizing for NB = 3 and can only fit 2 doubles per register on SSE, we unroll 6 times
  a_01 = _mm_loadu_pd(&a[0]); // GR
  a_12 = _mm_loadu_pd(&a[1]); // BG
  a_20 = _mm_loadh_pd(_mm_load_sd(&a[2]),
		      &a[0]); // RB

  a_0210 = _mm256_insertf128_pd(a_0210,a_01,0);
  a_0210 = _mm256_insertf128_pd(a_0210,a_20,1); // RBGR

  a_1021 = _mm256_insertf128_pd(a_1021,a_12,0);
  a_1021 = _mm256_insertf128_pd(a_1021,a_01,1); // GRBG

  a_2102 = _mm256_insertf128_pd(a_2102,a_20,0);
  a_2102 = _mm256_insertf128_pd(a_2102,a_12,1); // BGRB

  b_01 = _mm_loadu_pd(&b[0]); // GR
  b_12 = _mm_loadu_pd(&b[1]); // BG
  b_20 = _mm_loadh_pd(_mm_load_sd(&b[2]),
		      &b[0]); // RG

  b_0210 = _mm256_insertf128_pd(b_0210,b_01,0);
  b_0210 = _mm256_insertf128_pd(b_0210,b_20,1); // RBGR

  b_1021 = _mm256_insertf128_pd(b_1021,b_12,0);
  b_1021 = _mm256_insertf128_pd(b_1021,b_01,1); // GRBG

  b_2102 = _mm256_insertf128_pd(b_2102,b_20,0);
  b_2102 = _mm256_insertf128_pd(b_2102,b_12,1); // BGRB

  i = 0;
  x = 0;

  while ((x + 4) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)
    p_1 = _mm256_cvtps_pd(_mm_loadu_ps(&p[i])); // R2B1G1R1
    p_2 = _mm256_cvtps_pd(_mm_loadu_ps(&p[i+4])); // G3R3B2G2
    p_3 = _mm256_cvtps_pd(_mm_loadu_ps(&p[i+8])); // B4G4R4B3

    p_1 = _mm256_add_pd(_mm256_mul_pd(p_1,a_0210),b_0210); // (R2B1G1R1 * A0A2A1A0) + B0B2B1B0
    p_2 = _mm256_add_pd(_mm256_mul_pd(p_2,a_1021),b_1021); // (G3R3B2G2 * A1A0A2A1) + B1B0B2B1
    p_3 = _mm256_add_pd(_mm256_mul_pd(p_3,a_2102),b_2102); // (B4G4R4B3 * A2A1A0A2) + B2B1B0B2

    _mm_storeu_ps(&q[i],_mm256_cvtpd_ps(p_1));
    _mm_storeu_ps(&q[i+4],_mm256_cvtpd_ps(p_2));
    _mm_storeu_ps(&q[i+8],_mm256_cvtpd_ps(p_3));

    i += 12;
    x += 4;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (float) p[i]) + b[0];
    q[i+1] = (a[1] * (float) p[i+1]) + b[1];
    q[i+2] = (a[2] * (float) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_float(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}
#endif // PARSEC_USE_AVX


#ifdef PARSEC_USE_AVX512

// JMCG Vectorization - LOOPN for UCHAR (IN = unsigned char, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_avx512_uchar(double *a, double *b, int width, PEL *in, PEL *out) {
  unsigned char *p = (unsigned char *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m512d _a, _b;
  __m512d a_1, a_2, a_3, b_1, b_2, b_3;
  __m512d p_1,p_2,p_3;
  __m256i temp_l;
  int i, x;

  // We are only vectorizing for NB = 3, we are using a different approach for AVX512, with less instructions but I'm not sure about the latency
  _a = _mm512_mask_loadu_pd(_a,0b00000111, &a[0]); // TRASH + BGR
  _b = _mm512_mask_loadu_pd(_b,0b00000111, &b[0]); // TRASH + BGR

  a_1 = _mm512_permutexvar_pd(_mm512_set_epi64(1,0,2,1,0,2,1,0), _a); // a_1 = GRBGRBGR
  a_2 = _mm512_permutexvar_pd(_mm512_set_epi64(0,2,1,0,2,1,0,2), _a); // a_2 = RGBRBGRB
  a_3 = _mm512_permutexvar_pd(_mm512_set_epi64(2,1,0,2,1,0,2,1), _a); // a_3 = BGRBGRBG

  b_1 = _mm512_permutexvar_pd(_mm512_set_epi64(1,0,2,1,0,2,1,0), _b); // b_1 = GRBGRBGR
  b_2 = _mm512_permutexvar_pd(_mm512_set_epi64(0,2,1,0,2,1,0,2), _b); // b_2 = RGBRGBRB
  b_3 = _mm512_permutexvar_pd(_mm512_set_epi64(2,1,0,2,1,0,2,1), _b); // b_3 = BGRBGRBG

  i = 0;
  x = 0;

  while ((x + 8) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)

    temp_l = _mm256_cvtepu8_epi32(*(__m128i *) (&p[i]));
    p_1 = _mm512_cvtepi32_pd(temp_l); // G3R3B2G2 R2B1G1R1

    temp_l = _mm256_cvtepu8_epi32(*(__m128i *) (&p[i+8]));
    p_2 = _mm512_cvtepi32_pd(temp_l); // R6B5G5R5 B4G4R4B3

    temp_l = _mm256_cvtepu8_epi32(*(__m128i *) (&p[i+16]));
    p_3 = _mm512_cvtepi32_pd(temp_l); // B8G8R8B7 G7R7B6G6

    p_1 = _mm512_add_pd(_mm512_mul_pd(p_1,a_1),b_1);
    p_2 = _mm512_add_pd(_mm512_mul_pd(p_2,a_2),b_2);
    p_3 = _mm512_add_pd(_mm512_mul_pd(p_3,a_3),b_3);

    _mm256_storeu_ps(&q[i],_mm512_cvtpd_ps(p_1));
    _mm256_storeu_ps(&q[i+8],_mm512_cvtpd_ps(p_2));
    _mm256_storeu_ps(&q[i+16],_mm512_cvtpd_ps(p_3));

    i += 24;
    x += 8;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (unsigned char) p[i]) + b[0];
    q[i+1] = (a[1] * (unsigned char) p[i+1]) + b[1];
    q[i+2] = (a[2] * (unsigned char) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_uchar(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Uchar Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff UChar = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

// JMCG Vectorization - LOOPN for FLOAT (IN = float, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_avx512_float(double *a, double *b, int width, PEL *in, PEL *out) {
  float *p = (float *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m512d _a, _b;
  __m512d a_1, a_2, a_3, b_1, b_2, b_3;
  __m512d p_1,p_2,p_3;
  int i, x;

  // We are only vectorizing for NB = 3, we are using a different approach for AVX512, with less instructions but I'm not sure about the latency
  _a = _mm512_mask_loadu_pd(_a,0b00000111, &a[0]); // TRASH + BGR
  _b = _mm512_mask_loadu_pd(_b,0b00000111, &b[0]); // TRASH + BGR

  a_1 = _mm512_permutexvar_pd(_mm512_set_epi64(1,0,2,1,0,2,1,0), _a); // a_1 = GRBGRBGR
  a_2 = _mm512_permutexvar_pd(_mm512_set_epi64(0,2,1,0,2,1,0,2), _a); // a_2 = RGBRBGRB
  a_3 = _mm512_permutexvar_pd(_mm512_set_epi64(2,1,0,2,1,0,2,1), _a); // a_3 = BGRBGRBG

  b_1 = _mm512_permutexvar_pd(_mm512_set_epi64(1,0,2,1,0,2,1,0), _b); // b_1 = GRBGRBGR
  b_2 = _mm512_permutexvar_pd(_mm512_set_epi64(0,2,1,0,2,1,0,2), _b); // b_2 = RGBRGBRB
  b_3 = _mm512_permutexvar_pd(_mm512_set_epi64(2,1,0,2,1,0,2,1), _b); // b_3 = BGRBGRBG

  i = 0;
  x = 0;

  while ((x + 8) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)
    p_1 = _mm512_cvtps_pd(_mm256_loadu_ps(&p[i])); // G3R3B2G2 R2B1G1R1
    p_2 = _mm512_cvtps_pd(_mm256_loadu_ps(&p[i+8])); // R6B5G5R5 B4G4R4B3
    p_3 = _mm512_cvtps_pd(_mm256_loadu_ps(&p[i+16])); // B8G8R8B7 G7R7B6G6

    p_1 = _mm512_add_pd(_mm512_mul_pd(p_1,a_1),b_1);
    p_2 = _mm512_add_pd(_mm512_mul_pd(p_2,a_2),b_2);
    p_3 = _mm512_add_pd(_mm512_mul_pd(p_3,a_3),b_3);

    _mm256_storeu_ps(&q[i],_mm512_cvtpd_ps(p_1));
    _mm256_storeu_ps(&q[i+8],_mm512_cvtpd_ps(p_2));
    _mm256_storeu_ps(&q[i+16],_mm512_cvtpd_ps(p_3));

    i += 24;
    x += 8;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (float) p[i]) + b[0];
    q[i+1] = (a[1] * (float) p[i+1]) + b[1];
    q[i+2] = (a[2] * (float) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_float(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}
#endif // PARSEC_USE_AVX512


#ifdef PARSEC_USE_SSE
// JMCG Vectorization - LOOPN for UCHAR (IN = unsigned char, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_sse_uchar(double *a, double *b, int width, PEL *in, PEL *out) {
  unsigned char *p = (unsigned char *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m128d a_01, a_20, a_12, b_01, b_20, b_12;
  __m128d p_1,p_2,p_3,p_4,p_5,p_6;
  __m128i temp_l,temp_h;
  int i, x;

  // JMCG
  // Since we are only vectorizing for NB = 3 and can only fit 2 doubles per register on SSE
  // we unrolled 12 times to minimized loads of uchars, but in this final version,
  // as we use *(__m128i *) (&p[i]) is not really necessary, we could unroll 6 times only as we do for floats

  a_01 = _mm_loadu_pd(&a[0]);
  a_12 = _mm_loadu_pd(&a[1]);
  a_20 = _mm_loadh_pd(_mm_load_sd(&a[2]),
		      &a[0]);

  b_01 = _mm_loadu_pd(&b[0]);
  b_12 = _mm_loadu_pd(&b[1]);
  b_20 = _mm_loadh_pd(_mm_load_sd(&b[2]),
		      &b[0]);
  i = 0;
  x = 0;

  while ((x + 4) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i]));
    p_1 = _mm_cvtepi32_pd(temp_l); // G1R1
    temp_h = _mm_shuffle_epi32(temp_l,_MM_SHUFFLE(1,0,3,2));
    p_2 = _mm_cvtepi32_pd(temp_h); // R2B1

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i+4]));
    p_3 = _mm_cvtepi32_pd(temp_l); // B2G2
    temp_h = _mm_shuffle_epi32(temp_l,_MM_SHUFFLE(1,0,3,2));
    p_4 = _mm_cvtepi32_pd(temp_h); // G3R3

    temp_l = _mm_cvtepu8_epi32(*(__m128i *) (&p[i+8]));
    p_5 = _mm_cvtepi32_pd(temp_l); // R4B3
    temp_h = _mm_shuffle_epi32(temp_l,_MM_SHUFFLE(1,0,3,2));
    p_6 = _mm_cvtepi32_pd(temp_h); // G4R4


    p_1 = _mm_add_pd(_mm_mul_pd(p_1,a_01),b_01); // (G1R1 * A1A0) + B1B0
    p_2 = _mm_add_pd(_mm_mul_pd(p_2,a_20),b_20); // (R2B1 * A0A2) + B0B2
    p_3 = _mm_add_pd(_mm_mul_pd(p_3,a_12),b_12); // (B2G2 * A2A1) + B2B1

    p_4 = _mm_add_pd(_mm_mul_pd(p_4,a_01),b_01); // (G3R3 * A1A0) + B1B0
    p_5 = _mm_add_pd(_mm_mul_pd(p_5,a_20),b_20); // (R4B3 * A0A2) + B0B2
    p_6 = _mm_add_pd(_mm_mul_pd(p_6,a_12),b_12); // (B4G4 * A2A1) + B2B1

    _mm_storel_pi((__m64*) &q[i],_mm_cvtpd_ps(p_1));
    _mm_storel_pi((__m64*) &q[i+2],_mm_cvtpd_ps(p_2));
    _mm_storel_pi((__m64*) &q[i+4],_mm_cvtpd_ps(p_3));
    _mm_storel_pi((__m64*) &q[i+6],_mm_cvtpd_ps(p_4));
    _mm_storel_pi((__m64*) &q[i+8],_mm_cvtpd_ps(p_5));
    _mm_storel_pi((__m64*) &q[i+10],_mm_cvtpd_ps(p_6));

    i += 12;
    x += 4;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (unsigned char) p[i]) + b[0];
    q[i+1] = (a[1] * (unsigned char) p[i+1]) + b[1];
    q[i+2] = (a[2] * (unsigned char) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_uchar(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Uchar Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff UChar = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

// JMCG Vectorization - LOOPN for FLOAT (IN = float, OUT = float)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_sse_float(double *a, double *b, int width, PEL *in, PEL *out) {
  float *p = (float *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  __m128d a_01, a_20, a_12, b_01, b_20, b_12;
  __m128d p_1,p_2,p_3;
  __m128 temp;
  int i, x;

  // Since we are only vectorizing for NB = 3 and can only fit 2 doubles per register on SSE, we unroll 6 times
  a_01 = _mm_loadu_pd(&a[0]);
  a_12 = _mm_loadu_pd(&a[1]);
  a_20 = _mm_loadh_pd(_mm_load_sd(&a[2]),
		      &a[0]);

  b_01 = _mm_loadu_pd(&b[0]);
  b_12 = _mm_loadu_pd(&b[1]);
  b_20 = _mm_loadh_pd(_mm_load_sd(&b[2]),
		      &b[0]);
  i = 0;
  x = 0;

  while ((x + 2) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)
    p_1 = _mm_cvtps_pd(_mm_loadl_pi(temp, (__m64*) &p[i])); // G1R1
    p_2 = _mm_cvtps_pd(_mm_loadl_pi(temp, (__m64*) &p[i+2])); // R2B1
    p_3 = _mm_cvtps_pd(_mm_loadl_pi(temp, (__m64*) &p[i+4])); // B2G2

    p_1 = _mm_add_pd(_mm_mul_pd(p_1,a_01),b_01); // (G1R1 * A1A0) + B1B0
    p_2 = _mm_add_pd(_mm_mul_pd(p_2,a_20),b_20); // (R2B1 * A0A2) + B0B2
    p_3 = _mm_add_pd(_mm_mul_pd(p_3,a_12),b_12); // (B2G2 * A2A1) + B2B1

    _mm_storel_pi((__m64*) &q[i],_mm_cvtpd_ps(p_1));
    _mm_storel_pi((__m64*) &q[i+2],_mm_cvtpd_ps(p_2));
    _mm_storel_pi((__m64*) &q[i+4],_mm_cvtpd_ps(p_3));

    i += 6;
    x += 2;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (float) p[i]) + b[0];
    q[i+1] = (a[1] * (float) p[i+1]) + b[1];
    q[i+2] = (a[2] * (float) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_float(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}
#endif // PARSEC_USE_SSE

#ifdef PARSEC_USE_NEON

// JMCG Vectorization - LOOPN for UCHAR (IN = unsigned char, OUT = float) NOTE: NEON DOES NOT SUPPORT DOUBLE AT THIS POINT (Aug. 2013)
// We use inline functions instead of macros for debugging purposes

static void inline loopn_neon_uchar(double *a, double *b, int width, PEL *in, PEL *out) {
  unsigned char *p = (unsigned char *) in;
  float *q = (float *) out;

  // a and b are doubles, not supported by NEON, using floats
  float32x4_t a_0210, a_1021, a_2102, b_0210, b_1021, b_2102;
  float32x4_t p_1,p_2,p_3;
  int i, x;

  // JMCG Manual unroll
  // I cannot figure out exactly how to manually load doubles in NEON
  // converting to float, so we let gcc handle the conversion
  a_0210 = _MM_SETM((float)a[0],(float)a[2],(float)a[1],(float)a[0]); // RBGR
  a_1021 = _MM_SETM((float)a[1],(float)a[0],(float)a[2],(float)a[1]); // GRBG
  a_2102 = _MM_SETM((float)a[2],(float)a[1],(float)a[0],(float)a[2]); // BGRB

  b_0210 = _MM_SETM((float)b[0],(float)b[2],(float)b[1],(float)b[0]); // RBGR
  b_1021 = _MM_SETM((float)b[1],(float)b[0],(float)b[2],(float)b[1]); // GRBG
  b_2102 = _MM_SETM((float)b[2],(float)b[1],(float)b[0],(float)b[2]); // BGRB

  i = 0;
  x = 0;

  while ((x + 4) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)

    uint32x2_t load_temp;
    load_temp = vld1_lane_u32((uint32_t *)&p[i],load_temp,0); // Load 4 uchar values in the lower part of a 64 register
    p_1 = vcvtq_f32_u32(vmovl_u16(vreinterpret_u16_u32(vget_low_u32(vreinterpretq_u32_u16(vmovl_u8(vreinterpret_u8_u32(load_temp))))))); // expand the 4 uchars to fill all lanes of a 128 register (equivalent to Intel _mm_cvtepu8_epi32) // R2B1G1R1

    load_temp = vld1_lane_u32((uint32_t *)&p[i+4],load_temp,0);
    p_2 = vcvtq_f32_u32(vmovl_u16(vreinterpret_u16_u32(vget_low_u32(vreinterpretq_u32_u16(vmovl_u8(vreinterpret_u8_u32(load_temp))))))); // G3R3B2G2

    load_temp = vld1_lane_u32((uint32_t *)&p[i+8],load_temp,0);
    p_3 = vcvtq_f32_u32(vmovl_u16(vreinterpret_u16_u32(vget_low_u32(vreinterpretq_u32_u16(vmovl_u8(vreinterpret_u8_u32(load_temp))))))); // B4G4R4B3

    p_1 = vaddq_f32(vmulq_f32(p_1,a_0210),b_0210); // (R2B1G1R1 * A0A2A1A0) + B0B2B1B0
    p_2 = vaddq_f32(vmulq_f32(p_2,a_1021),b_1021); // (G3R3B2G2 * A1A0A2A1) + B1B0B2B1
    p_3 = vaddq_f32(vmulq_f32(p_3,a_2102),b_2102); // (B4G4R4B3 * A2A1A0A2) + B2B1B0B2

    vst1q_f32(&q[i],p_1);
    vst1q_f32(&q[i+4],p_2);
    vst1q_f32(&q[i+8],p_3);

    i += 12;
    x += 4;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (unsigned char) p[i]) + b[0];
    q[i+1] = (a[1] * (unsigned char) p[i+1]) + b[1];
    q[i+2] = (a[2] * (unsigned char) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_uchar(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //printf("Uchar Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff UChar = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

// JMCG Vectorization - LOOPN for FLOAT (IN = float, OUT = float) NOTE: NEON DOES NOT SUPPORT DOUBLE AT THIS POINT (Aug. 2013)
// We use inline functions instead of macros for debugging purposes
static void inline loopn_neon_float(double *a, double *b, int width, PEL *in, PEL *out) {
  float *p = (float *) in;
  float *q = (float *) out;

  // a and b are doubles, so all calculations are done over doubles as GCC would do with single data
  float32x4_t a_0210, a_1021, a_2102, b_0210, b_1021, b_2102;
  float32x4_t p_1,p_2,p_3;

  int i, x;

  // JMCG Manual unroll
  // I cannot figure out exactly how to manually load doubles in NEON
  // converting to float, so we let gcc handle the conversion
  a_0210 = _MM_SETM((float)a[0],(float)a[2],(float)a[1],(float)a[0]); // RBGR
  a_1021 = _MM_SETM((float)a[1],(float)a[0],(float)a[2],(float)a[1]); // GRBG
  a_2102 = _MM_SETM((float)a[2],(float)a[1],(float)a[0],(float)a[2]); // BGRB

  b_0210 = _MM_SETM((float)b[0],(float)b[2],(float)b[1],(float)b[0]); // RBGR
  b_1021 = _MM_SETM((float)b[1],(float)b[0],(float)b[2],(float)b[1]); // GRBG
  b_2102 = _MM_SETM((float)b[2],(float)b[1],(float)b[0],(float)b[2]); // BGRB

  i = 0;
  x = 0;

  while ((x + 4) < width) {
    // Most likely (for parsec native width is 79, 68 and 25)

    p_1 = vld1q_f32(&p[i]); // R2B1G1R1
    p_2 = vld1q_f32(&p[i+4]); // G3R3B2G2
    p_3 = vld1q_f32(&p[i+8]); // B4G4R4B3

    p_1 = vaddq_f32(vmulq_f32(p_1,a_0210),b_0210); // (R2B1G1R1 * A0A2A1A0) + B0B2B1B0
    p_2 = vaddq_f32(vmulq_f32(p_2,a_1021),b_1021); // (G3R3B2G2 * A1A0A2A1) + B1B0B2B1
    p_3 = vaddq_f32(vmulq_f32(p_3,a_2102),b_2102); // (B4G4R4B3 * A2A1A0A2) + B2B1B0B2

    vst1q_f32(&q[i],p_1);
    vst1q_f32(&q[i+4],p_2);
    vst1q_f32(&q[i+8],p_3);

    i += 12;
    x += 4;
  }

  // Compute leftovers
  for( ; x < width; x++ ) {
    q[i] = (a[0] * (float) p[i]) + b[0];
    q[i+1] = (a[1] * (float) p[i+1]) + b[1];
    q[i+2] = (a[2] * (float) p[i+2]) + b[2];
    i+=3;
  }
#ifdef DEBUG_SIMD
  float test_out[width*3];
  validate_loopn_float(a, b, width, p, &test_out[0]);
  i = 0;
  for(x = 0 ; x < width; x++ ) {
    //    printf("Q[0] %f Test[0] %f Q[1] %f Test[1] %f Q[2] %f Test[2] %f\n",q[i],test_out[i],q[i+1],test_out[i+1],q[i+2],test_out[i+2]); fflush(stdout);
    diff_l = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff_l > max_diff_l) {
      max_diff_l = diff_l;
      printf("Maxdiff = %f\n",max_diff_l);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

#endif // PARSEC_USE_NEON

/* JMCG END */


/* Non-complex input, any output, all bands of the constant equal.
 */
#define LOOP1( IN, OUT ) { \
	IN * restrict p = (IN *) in[0]; \
	OUT * restrict q = (OUT *) out; \
	OUT a1 = a[0]; \
	OUT b1 = b[0]; \
	int sz = width * nb; \
	\
	for( x = 0; x < sz; x++ ) \
		q[x] = a1 * (OUT) p[x] + b1; \
}

/* Non-complex input, any output.
 */
#define LOOPN( IN, OUT ) { \
	IN * restrict p = (IN *) in[0]; \
	OUT * restrict q = (OUT *) out; \
	\
	for( i = 0, x = 0; x < width; x++ ) \
		for( k = 0; k < nb; k++, i++ ) \
			q[i] = a[k] * (OUT) p[i] + b[k]; \
}

#define LOOP( IN, OUT ) { \
	if( linear->a->n == 1 && linear->b->n == 1 ) { \
		LOOP1( IN, OUT ); \
	} \
	else { \
		LOOPN( IN, OUT ); \
	} \
}

/* Complex input, complex output.
 */
#define LOOPCMPLXN( IN, OUT ) { \
	IN * restrict p = (IN *) in[0]; \
	OUT * restrict q = (OUT *) out; \
	\
	for( x = 0; x < width; x++ ) \
		for( k = 0; k < nb; k++ ) { \
			q[0] = a[k] * p[0] + b[k]; \
			q[1] = p[1]; \
			q += 2; \
			p += 2; \
		} \
}

/* Non-complex input, any output, all bands of the constant equal, uchar
 * output.
 */
#define LOOP1uc( IN ) { \
	IN * restrict p = (IN *) in[0]; \
	VipsPel * restrict q = (VipsPel *) out; \
	float a1 = a[0]; \
	float b1 = b[0]; \
	int sz = width * nb; \
	\
	for( x = 0; x < sz; x++ ) { \
		float t = a1 * p[x] + b1; \
		\
		q[x] = VIPS_FCLIP( 0, t, 255 ); \
	} \
}

/* Non-complex input, uchar output.
 */
#define LOOPNuc( IN ) { \
	IN * restrict p = (IN *) in[0]; \
	VipsPel * restrict q = (VipsPel *) out; \
	\
	for( i = 0, x = 0; x < width; x++ ) \
		for( k = 0; k < nb; k++, i++ ) { \
			double t = a[k] * p[i] + b[k]; \
			\
			q[i] = VIPS_FCLIP( 0, t, 255 ); \
		} \
}

#define LOOPuc( IN ) { \
	if( linear->a->n == 1 && linear->b->n == 1 ) { \
		LOOP1uc( IN ); \
	} \
	else { \
		LOOPNuc( IN ); \
	} \
}

/* Complex input, uchar output.
 */
#define LOOPCMPLXNuc( IN ) { \
	IN * restrict p = (IN *) in[0]; \
	VipsPel * restrict q = (VipsPel *) out; \
	\
	for( i = 0, x = 0; x < width; x++ ) \
		for( k = 0; k < nb; k++, i++ ) { \
			double t = a[k] * p[0] + b[k]; \
			\
			q[i] = VIPS_FCLIP( 0, t, 255 ); \
			p += 2; \
		} \
}

/* Lintra a buffer, n set of scale/offset.
 */
static void
vips_linear_buffer( VipsArithmetic *arithmetic,
	VipsPel *out, VipsPel **in, int width )
{
	VipsImage *im = arithmetic->ready[0];
	VipsLinear *linear = (VipsLinear *) arithmetic;
	double * restrict a = linear->a_ready;
	double * restrict b = linear->b_ready;
	int nb = im->Bands;

	int i, x, k;

	if( linear->uchar ) {
	  switch( vips_image_get_format( im ) ) {
	  case VIPS_FORMAT_UCHAR: 	    LOOPuc( unsigned char ); break;
	  case VIPS_FORMAT_CHAR:	    LOOPuc( signed char ); break;
	  case VIPS_FORMAT_USHORT: 	    LOOPuc( unsigned short ); break;
	  case VIPS_FORMAT_SHORT: 	    LOOPuc( signed short ); break;
	  case VIPS_FORMAT_UINT: 	    LOOPuc( unsigned int ); break;
	  case VIPS_FORMAT_INT:	            LOOPuc( signed int );  break;
	  case VIPS_FORMAT_FLOAT:	    LOOPuc( float ); break;
	  case VIPS_FORMAT_DOUBLE: 	    LOOPuc( double ); break;
	  case VIPS_FORMAT_COMPLEX: 	    LOOPCMPLXNuc( float ); break;
	  case VIPS_FORMAT_DPCOMPLEX: 	    LOOPCMPLXNuc( double ); break;

	  default:
	    g_assert_not_reached();
	  }
	} else {
	  if(nb==3) {
	    switch( vips_image_get_format( im ) ) {
	      /* JMCG BEGIN */
#ifdef SIMD_WIDTH
#ifdef PARSEC_USE_SSE
	    case VIPS_FORMAT_UCHAR:	      loopn_sse_uchar(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_AVX
	    case VIPS_FORMAT_UCHAR:           loopn_avx_uchar(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_AVX512
	    case VIPS_FORMAT_UCHAR:           loopn_avx512_uchar(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_NEON
	    case VIPS_FORMAT_UCHAR:           loopn_neon_uchar(a, b, width, in[0], out); break;
#endif
#else
	    case VIPS_FORMAT_UCHAR:	      LOOP( unsigned char, float ); break;
#endif
	      /* JMCG END */
	    case VIPS_FORMAT_CHAR:	      LOOP( signed char, float ); break;
	    case VIPS_FORMAT_USHORT:	      LOOP( unsigned short, float ); break;
	    case VIPS_FORMAT_SHORT:	      LOOP( signed short, float ); break;
	    case VIPS_FORMAT_UINT:	      LOOP( unsigned int, float ); break;
	    case VIPS_FORMAT_INT:	      LOOP( signed int, float );  break;
	      /* JMCG BEGIN */
#ifdef SIMD_WIDTH
#ifdef PARSEC_USE_SSE
	    case VIPS_FORMAT_FLOAT:  	      loopn_sse_float(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_AVX
	    case VIPS_FORMAT_FLOAT:           loopn_avx_float(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_AVX512
	    case VIPS_FORMAT_FLOAT:           loopn_avx512_float(a, b, width, in[0], out); break;
#endif
#ifdef PARSEC_USE_NEON
	    case VIPS_FORMAT_FLOAT:	      loopn_neon_float(a, b, width, in[0], out); break;
#endif
#else
	    case VIPS_FORMAT_FLOAT:   	      LOOP( float, float ); break;
#endif
	      /* JMCG END */
	    case VIPS_FORMAT_DOUBLE: 	      LOOP( double, double ); break;
	    case VIPS_FORMAT_COMPLEX:	      LOOPCMPLXN( float, float ); break;
	    case VIPS_FORMAT_DPCOMPLEX:	      LOOPCMPLXN( double, double ); break;

	    default:
	      g_assert_not_reached();
	    }
	  } else {
	    switch( vips_image_get_format( im ) ) {
	    case VIPS_FORMAT_UCHAR:	      LOOP( unsigned char, float ); break;
	    case VIPS_FORMAT_CHAR:	      LOOP( signed char, float ); break;
	    case VIPS_FORMAT_USHORT:	      LOOP( unsigned short, float ); break;
	    case VIPS_FORMAT_SHORT:	      LOOP( signed short, float ); break;
	    case VIPS_FORMAT_UINT:	      LOOP( unsigned int, float ); break;
	    case VIPS_FORMAT_INT:	      LOOP( signed int, float );  break;
	    case VIPS_FORMAT_FLOAT:    	      LOOP( float, float ); break;
	    case VIPS_FORMAT_DOUBLE:	      LOOP( double, double ); break;
	    case VIPS_FORMAT_COMPLEX:	      LOOPCMPLXN( float, float ); break;
	    case VIPS_FORMAT_DPCOMPLEX:	      LOOPCMPLXN( double, double ); break;
	    default:
	      g_assert_not_reached();
	    }
	  }
	}
}
/* Save a bit of typing.
 */
#define UC VIPS_FORMAT_UCHAR
#define C VIPS_FORMAT_CHAR
#define US VIPS_FORMAT_USHORT
#define S VIPS_FORMAT_SHORT
#define UI VIPS_FORMAT_UINT
#define I VIPS_FORMAT_INT
#define F VIPS_FORMAT_FLOAT
#define X VIPS_FORMAT_COMPLEX
#define D VIPS_FORMAT_DOUBLE
#define DX VIPS_FORMAT_DPCOMPLEX

/* Format doesn't change with linear.
 */
static const VipsBandFormat vips_linear_format_table[10] = {
/* UC  C   US  S   UI  I   F   X   D   DX */
   F,  F,  F,  F,  F,  F,  F,  X,  D,  DX
};

static void
vips_linear_class_init( VipsLinearClass *class )
{
	GObjectClass *gobject_class = G_OBJECT_CLASS( class );
	VipsObjectClass *object_class = (VipsObjectClass *) class;
	VipsArithmeticClass *aclass = VIPS_ARITHMETIC_CLASS( class );

	gobject_class->set_property = vips_object_set_property;
	gobject_class->get_property = vips_object_get_property;

	object_class->nickname = "linear";
	object_class->description = _( "calculate (a * in + b)" );
	object_class->build = vips_linear_build;

	aclass->process_line = vips_linear_buffer;

	vips_arithmetic_set_format_table( aclass, vips_linear_format_table );

	VIPS_ARG_BOXED( class, "a", 110,
		_( "a" ),
		_( "Multiply by this" ),
		VIPS_ARGUMENT_REQUIRED_INPUT,
		G_STRUCT_OFFSET( VipsLinear, a ),
		VIPS_TYPE_ARRAY_DOUBLE );

	VIPS_ARG_BOXED( class, "b", 111,
		_( "b" ),
		_( "Add this" ),
		VIPS_ARGUMENT_REQUIRED_INPUT,
		G_STRUCT_OFFSET( VipsLinear, b ),
		VIPS_TYPE_ARRAY_DOUBLE );

	VIPS_ARG_BOOL( class, "uchar", 112,
		_( "uchar" ),
		_( "Output should be uchar" ),
		VIPS_ARGUMENT_OPTIONAL_INPUT,
		G_STRUCT_OFFSET( VipsLinear, uchar ),
		FALSE );

}

static void
vips_linear_init( VipsLinear *linear )
{
}

static int
vips_linearv( VipsImage *in, VipsImage **out,
	double *a, double *b, int n, va_list ap )
{
	VipsArea *area_a;
	VipsArea *area_b;
	int result;

	area_a = (VipsArea *) vips_array_double_new( a, n );
	area_b = (VipsArea *) vips_array_double_new( b, n );

	result = vips_call_split( "linear", ap, in, out, area_a, area_b );

	vips_area_unref( area_a );
	vips_area_unref( area_b );

	return( result );
}

/**
 * vips_linear:
 * @in: image to transform
 * @out: output image
 * @a: (array length=n): array of constants for multiplication
 * @b: (array length=n): array of constants for addition
 * @n: length of constant arrays
 * @...: %NULL-terminated list of optional named arguments
 *
 * Optional arguments:
 *
 * @uchar: output uchar pixels
 *
 * Pass an image through a linear transform, ie. (@out = @in * @a + @b). Output
 * is float for integer input, double for double input, complex for
 * complex input and double complex for double complex input. Set @uchar to
 * output uchar pixels.
 *
 * If the arrays of constants have just one element, that constant is used for
 * all image bands. If the arrays have more than one element and they have
 * the same number of elements as there are bands in the image, then
 * one array element is used for each band. If the arrays have more than one
 * element and the image only has a single band, the result is a many-band
 * image where each band corresponds to one array element.
 *
 * See also: vips_linear1(), vips_add().
 *
 * Returns: 0 on success, -1 on error
 */
int
vips_linear( VipsImage *in, VipsImage **out, double *a, double *b, int n, ... )
{
	va_list ap;
	int result;

	va_start( ap, n );
	result = vips_linearv( in, out, a, b, n, ap );
	va_end( ap );

	return( result );
}

/**
 * vips_linear1:
 * @in: image to transform
 * @out: output image
 * @a: constant for multiplication
 * @b: constant for addition
 * @...: %NULL-terminated list of optional named arguments
 *
 * Optional arguments:
 *
 * @uchar: output uchar pixels
 *
 * Run vips_linear() with a single constant.
 *
 * See also: vips_linear().
 *
 * Returns: 0 on success, -1 on error
 */
int
vips_linear1( VipsImage *in, VipsImage **out, double a, double b, ... )
{
	va_list ap;
	int result;

	va_start( ap, b );
	result = vips_linearv( in, out, &a, &b, 1, ap );
	va_end( ap );

	return( result );
}
