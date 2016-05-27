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
 */

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

/*

    This file is part of VIPS.

    VIPS is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

 */

/*

    These files are distributed with VIPS - http://www.vips.ecs.soton.ac.uk

 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/
#include <vips/intl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <vips/vips.h>
#include <vips/internal.h>

#ifdef HAVE_LIBOIL
#include <liboil/liboil.h>
#endif /*HAVE_LIBOIL*/

#ifdef WITH_DMALLOC
#include <dmalloc.h>
#endif /*WITH_DMALLOC*/

/* JMCG BEGIN */
//#define DFTYPE
#include "simd_defines.h"

//#define DEBUG_SIMD
#ifdef DEBUG_SIMD
#include <math.h>
float diff = 0.0f;
float max_diff = 0.0f;
#endif
/* JMCG END */

/* Struct we need for im_generate().
 */
typedef struct {
	int n;			/* Number of bands of constants */
	double *a, *b;
} LintraInfo;

/* Define what we do for each band element type. Non-complex input, any
 * output.
 */
#define LOOP( IN, OUT ) { \
	IN *p = (IN *) in; \
	OUT *q = (OUT *) out; \
	\
	for( x = 0; x < sz; x++ ) \
		q[x] = a * (OUT) p[x] + b; \
}

/* Complex input, complex output.
 */
#define LOOPCMPLX( IN, OUT ) { \
	IN *p = (IN *) in; \
	OUT *q = (OUT *) out; \
	\
	for( x = 0; x < sz; x++ ) { \
		q[0] = a * p[0] + b; \
		q[1] = a * p[1]; \
		q += 2; \
		p += 2; \
	} \
}

#ifdef HAVE_LIBOIL
/* Process granularity.
 */
#define CHUNKS (1000)

/* d[] = s[] * b + c, with liboil
 */
static void
lintra_f32( float *d, float *s, int n, float b, float c )
{
	float buf[CHUNKS];
	int i;

	for( i = 0; i < n; i += CHUNKS ) {
		oil_scalarmultiply_f32_ns( buf, s,
			&b, IM_MIN( CHUNKS, n - i ) );
		oil_scalaradd_f32_ns( d, buf,
			&c, IM_MIN( CHUNKS, n - i ) );

		s += CHUNKS;
		d += CHUNKS;
	}
}
#endif /*HAVE_LIBOIL*/

/* Lintra a buffer, 1 set of scale/offset.
 */
static int
lintra1_gen( PEL *in, PEL *out, int width, IMAGE *im, LintraInfo *inf )
{
	double a = inf->a[0];
	double b = inf->b[0];
	int sz = width * im->Bands;
	int x;

	/* Lintra all input types.
         */
        switch( im->BandFmt ) {
        case IM_BANDFMT_UCHAR: 		LOOP( unsigned char, float ); break;
        case IM_BANDFMT_CHAR: 		LOOP( signed char, float ); break;
        case IM_BANDFMT_USHORT: 	LOOP( unsigned short, float ); break;
        case IM_BANDFMT_SHORT: 		LOOP( signed short, float ); break;
        case IM_BANDFMT_UINT: 		LOOP( unsigned int, float ); break;
        case IM_BANDFMT_INT: 		LOOP( signed int, float );  break;
        case IM_BANDFMT_FLOAT:
#ifdef HAVE_LIBOIL
		lintra_f32( (float *) out, (float *) in, sz, a, b );
#else /*!HAVE_LIBOIL*/
		LOOP( float, float );
#endif /*HAVE_LIBOIL*/
		break;

        case IM_BANDFMT_DOUBLE:		LOOP( double, double ); break;
        case IM_BANDFMT_COMPLEX:	LOOPCMPLX( float, float ); break;
        case IM_BANDFMT_DPCOMPLEX:	LOOPCMPLX( double, double ); break;

        default:
		assert( 0 );
        }

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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff UChar = %f\n",max_diff);
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
    temp = _mm_loadu_ps(&p[i]);
    p_2 = _mm256_cvtps_pd(_mm_loadu_ps(&p[i+4])); // G3R3B2G2
    temp = _mm_loadu_ps(&p[i+4]);
    p_3 = _mm256_cvtps_pd(_mm_loadu_ps(&p[i+8])); // B4G4R4B3
    temp = _mm_loadu_ps(&p[i+8]);

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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff = %f\n",max_diff);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}
#endif // PARSEC_USE_AVX


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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff UChar = %f\n",max_diff);
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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff = %f\n",max_diff);
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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff UChar = %f\n",max_diff);
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
    diff = fabs(q[i] - test_out[i]) + fabs(q[i+1] - test_out[i+1]) + fabs(q[i+2] - test_out[i+2]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff = %f\n",max_diff);
      fflush(stdout);
    }
    i += 3;
  }
#endif
}

#endif // PARSEC_USE_NEON

/* JMCG END */

/* Define what we do for each band element type. Non-complex input, any
 * output.
 */
#define LOOPN( IN, OUT ) {\
	IN *p = (IN *) in;\
	OUT *q = (OUT *) out;\
	\
	for( i = 0, x = 0; x < width; x++ ) {			\
	  for( k = 0; k < nb; k++, i++ ) {			\
	    q[i] = a[k] * (OUT) p[i] + b[k];			\
	  }							\
	}							\
  }

/* Complex input, complex output.
 */
#define LOOPCMPLXN( IN, OUT ) {\
	IN *p = (IN *) in;\
	OUT *q = (OUT *) out;\
	\
	for( x = 0; x < width; x++ ) \
		for( k = 0; k < nb; k++ ) {\
			q[0] = a[k] * p[0] + b[k];\
			q[1] = a[k] * p[1];\
			q += 2;\
			p += 2;\
		}\
}

/* Lintra a buffer, n set of scale/offset.
 */
static int
lintran_gen( PEL *in, PEL *out, int width, IMAGE *im, LintraInfo *inf )
{
	double *a = inf->a;
	double *b = inf->b;
	int nb = im->Bands;
	int i, x, k;

	/* Lintra all input types.
         */
        switch( im->BandFmt ) {
	  /* JMCG BEGIN */
#ifdef SIMD_WIDTH
#ifdef PARSEC_USE_SSE
	case IM_BANDFMT_UCHAR: 		assert(nb==3); loopn_sse_uchar(a, b, width, in, out); break;
#endif
#ifdef PARSEC_USE_AVX
	case IM_BANDFMT_UCHAR:          assert(nb==3); loopn_avx_uchar(a, b, width, in, out); break;
#endif
#ifdef PARSEC_USE_NEON
	case IM_BANDFMT_UCHAR:          assert(nb==3); loopn_neon_uchar(a, b, width, in, out); break;
#endif
#else
	case IM_BANDFMT_UCHAR: 		LOOPN( unsigned char, float ); break;
#endif
        case IM_BANDFMT_CHAR: 		LOOPN( signed char, float ); break;
        case IM_BANDFMT_USHORT: 	LOOPN( unsigned short, float ); break;
        case IM_BANDFMT_SHORT: 		LOOPN( signed short, float ); break;
        case IM_BANDFMT_UINT: 		LOOPN( unsigned int, float ); break;
        case IM_BANDFMT_INT: 		LOOPN( signed int, float );  break;
#ifdef SIMD_WIDTH
#ifdef PARSEC_USE_SSE
	case IM_BANDFMT_FLOAT: 		 assert(nb==3); loopn_sse_float(a, b, width, in, out); break;
#endif
#ifdef PARSEC_USE_AVX
	case IM_BANDFMT_FLOAT:           assert(nb==3); loopn_avx_float(a, b, width, in, out); break;
#endif
#ifdef PARSEC_USE_NEON
	case IM_BANDFMT_FLOAT: 		 assert(nb==3); loopn_neon_float(a, b, width, in, out); break;
#endif
#else
        case IM_BANDFMT_FLOAT: 		LOOPN( float, float ); break;
#endif
        case IM_BANDFMT_DOUBLE:		LOOPN( double, double ); break;
        case IM_BANDFMT_COMPLEX:	LOOPCMPLXN( float, float ); break;
        case IM_BANDFMT_DPCOMPLEX:	LOOPCMPLXN( double, double ); break;

        default:
		assert( 0 );
        }

	return( 0 );
}
/* JMCG END */
/* 1 band image, n band vector.
 */
#define LOOPNV( IN, OUT ) { \
	IN *p = (IN *) in; \
	OUT *q = (OUT *) out; \
	\
	for( i = 0, x = 0; x < width; x++ ) { \
		OUT v = p[x]; \
		\
		for( k = 0; k < nb; k++, i++ ) \
			q[i] = a[k] * v + b[k]; \
	} \
}

#define LOOPCMPLXNV( IN, OUT ) { \
	IN *p = (IN *) in; \
	OUT *q = (OUT *) out; \
	\
	for( x = 0; x < width; x++ ) { \
		OUT p0 = p[0]; \
		OUT p1 = p[1]; \
		\
		for( k = 0; k < nb; k++ ) { \
			q[0] = a[k] * p0 + b[k]; \
			q[1] = a[k] * p1; \
			q += 2; \
		} \
		\
		p += 2; \
	} \
}

static int
lintranv_gen( PEL *in, PEL *out, int width, IMAGE *im, LintraInfo *inf )
{
	double *a = inf->a;
	double *b = inf->b;
	int nb = inf->n;
	int i, x, k;

	/* Lintra all input types.
         */
        switch( im->BandFmt ) {
        case IM_BANDFMT_UCHAR: 		LOOPNV( unsigned char, float ); break;
        case IM_BANDFMT_CHAR: 		LOOPNV( signed char, float ); break;
        case IM_BANDFMT_USHORT: 	LOOPNV( unsigned short, float ); break;
        case IM_BANDFMT_SHORT: 		LOOPNV( signed short, float ); break;
        case IM_BANDFMT_UINT: 		LOOPNV( unsigned int, float ); break;
        case IM_BANDFMT_INT: 		LOOPNV( signed int, float );  break;
        case IM_BANDFMT_FLOAT: 		LOOPNV( float, float ); break;
        case IM_BANDFMT_DOUBLE:		LOOPNV( double, double ); break;
        case IM_BANDFMT_COMPLEX:	LOOPCMPLXNV( float, float ); break;
        case IM_BANDFMT_DPCOMPLEX:	LOOPCMPLXNV( double, double ); break;

        default:
		assert( 0 );
        }

	return( 0 );
}

/**
 * im_lintra_vec:
 * @n: array size
 * @a: array of constants for multiplication
 * @in: image to transform
 * @b: array of constants for addition
 * @out: output image
 *
 * Pass an image through a linear transform - ie. @out = @in * @a + @b. Output
 * is always float for integer input, double for double input, complex for
 * complex input and double complex for double complex input.
 *
 * If the arrays of constants have just one element, that constant are used for
 * all image bands. If the arrays have more than one element and they have
 * the same number of elements as there are bands in the image, then
 * one array element is used for each band. If the arrays have more than one
 * element and the image only has a single band, the result is a many-band
 * image where each band corresponds to one array element.
 *
 * See also: im_add(), im_lintra().
 *
 * Returns: 0 on success, -1 on error
 */
int
im_lintra_vec( int n, double *a, IMAGE *in, double *b, IMAGE *out )
{
	LintraInfo *inf;
	int i;

	if( im_piocheck( in, out ) ||
		im_check_vector( "im_lintra_vec", n, in ) ||
		im_check_uncoded( "lintra_vec", in ) )
		return( -1 );

	/* Prepare output header.
	 */
	if( im_cp_desc( out, in ) )
		return( -1 );
	if( vips_bandfmt_isint( in->BandFmt ) )
		out->BandFmt = IM_BANDFMT_FLOAT;
	if( in->Bands == 1 )
		out->Bands = n;

	/* Make space for a little buffer.
	 */
	if( !(inf = IM_NEW( out, LintraInfo )) ||
		!(inf->a = IM_ARRAY( out, n, double )) ||
		!(inf->b = IM_ARRAY( out, n, double )) )
		return( -1 );
	inf->n = n;
	for( i = 0; i < n; i++ ) {
		inf->a[i] = a[i];
		inf->b[i] = b[i];
	}

	/* Generate!
	 */
	if( n == 1 ) {
		if( im_wrapone( in, out,
			(im_wrapone_fn) lintra1_gen, in, inf ) )
			return( -1 );
	}
	else if( in->Bands == 1 ) {
		if( im_wrapone( in, out,
			(im_wrapone_fn) lintranv_gen, in, inf ) )
			return( -1 );
	}
	else {
		if( im_wrapone( in, out,
			(im_wrapone_fn) lintran_gen, in, inf ) )
			return( -1 );
	}

	return( 0 );
}

/**
 * im_lintra:
 * @a: constant for multiplication
 * @in: image to transform
 * @b: constant for addition
 * @out: output image
 *
 * Pass an image through a linear transform - ie. @out = @in * @a + @b. Output
 * is always float for integer input, double for double input, complex for
 * complex input and double complex for double complex input.
 *
 * See also: im_add(), im_lintra_vec().
 *
 * Returns: 0 on success, -1 on error
 */
int
im_lintra( double a, IMAGE *in, double b, IMAGE *out )
{
	return( im_lintra_vec( 1, &a, in, &b, out ) );
}
