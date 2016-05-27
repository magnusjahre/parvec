/* im_convsep
 *
 * Copyright: 1990, N. Dessipris.
 *
 * Author: Nicos Dessipris
 * Written on: 29/04/1991
 * Modified on: 29/4/93 K.Martinez  for Sys5
 * 9/3/01 JC
 *	- rewritten using im_conv()
 * 27/7/01 JC
 *	- rejects masks with scale == 0
 * 7/4/04
 *	- now uses im_embed() with edge stretching on the input, not
 *	  the output
 *	- sets Xoffset / Yoffset
 * 21/4/04
 *	- scale down int convolves at 1/2 way mark, much less likely to integer
 *	  overflow on intermediates
 * 12/5/08
 * 	- int rounding was +1 too much, argh
 * 3/2/10
 * 	- gtkdoc
 * 	- more cleanups
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
#include <limits.h>
#include <assert.h>

#include <vips/vips.h>

#ifdef WITH_DMALLOC
#include <dmalloc.h>
#endif /*WITH_DMALLOC*/

// JMCG
//#define DFTYPE
#include "simd_defines.h"


//#define DEBUG_SIMD
#ifdef DEBUG_SIMD
#include <math.h>
int diff = 0;
int max_diff = 0;
#endif


/* Our parameters ... we take a copy of the mask argument.
 */
typedef struct {
	IMAGE *in;
	IMAGE *out;
	INTMASK *mask;	/* Copy of mask arg */

	int size;	/* N for our 1xN or Nx1 mask */
	int scale;	/* Our scale ... we have to square mask->scale */

	int underflow;	/* Global underflow/overflow counts */
	int overflow;
} Conv;

/* End of evaluation --- print overflows and underflows.
 */
static int
conv_destroy( Conv *conv )
{
	/* Print underflow/overflow count.
	 */
	if( conv->overflow || conv->underflow )
		im_warn( "im_convsep", _( "%d overflows and %d underflows "
			"detected" ), conv->overflow, conv->underflow );

	if( conv->mask ) {
		(void) im_free_imask( conv->mask );
		conv->mask = NULL;
	}

        return( 0 );
}

static Conv *
conv_new( IMAGE *in, IMAGE *out, INTMASK *mask )
{
        Conv *conv = IM_NEW( out, Conv );

        if( !conv )
                return( NULL );

        conv->in = in;
        conv->out = out;
        conv->mask = NULL;
	conv->size = mask->xsize * mask->ysize;
	conv->scale = mask->scale * mask->scale;
        conv->underflow = 0;
        conv->overflow = 0;

        if( im_add_close_callback( out,
		(im_callback_fn) conv_destroy, conv, NULL ) ||
		!(conv->mask = im_dup_imask( mask, "conv_mask" )) )
                return( NULL );

        return( conv );
}

/* Our sequence value.
 */
typedef struct {
	Conv *conv;
	REGION *ir;		/* Input region */

	PEL *sum;		/* Line buffer */

	int underflow;		/* Underflow/overflow counts */
	int overflow;
} ConvSequence;

/* Free a sequence value.
 */
static int
conv_stop( void *vseq, void *a, void *b )
{
	ConvSequence *seq = (ConvSequence *) vseq;
	Conv *conv = (Conv *) b;

	/* Add local under/over counts to global counts.
	 */
	conv->overflow += seq->overflow;
	conv->underflow += seq->underflow;

	IM_FREEF( im_region_free, seq->ir );

	return( 0 );
}

/* Convolution start function.
 */
static void *
conv_start( IMAGE *out, void *a, void *b )
{
	IMAGE *in = (IMAGE *) a;
	Conv *conv = (Conv *) b;
	ConvSequence *seq;

	if( !(seq = IM_NEW( out, ConvSequence )) )
		return( NULL );

	/* Init!
	 */
	seq->conv = conv;
	seq->ir = NULL;
	seq->sum = NULL;
	seq->underflow = 0;
	seq->overflow = 0;

	/* Attach region and arrays.
	 */
	seq->ir = im_region_create( in );
	if( vips_bandfmt_isint( conv->out->BandFmt ) )
		seq->sum = (PEL *)
			IM_ARRAY( out, IM_IMAGE_N_ELEMENTS( in ), int );
	else
		seq->sum = (PEL *)
			IM_ARRAY( out, IM_IMAGE_N_ELEMENTS( in ), double );
	if( !seq->ir || !seq->sum ) {
		conv_stop( seq, in, conv );
		return( NULL );
	}

	return( (void *) seq );
}


/* What we do for every point in the mask, for each pixel.
 */
//#define VERTICAL_CONV { z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];printf("Z %d Li %d Coeff[z]: %d vFrom[li]: %d\n",z,li,coeff[z],vfrom[li]); fflush(stdout); }
#define VERTICAL_CONV { z -= 1; li -= lskip; sum += coeff[z] * vfrom[li]; }
#define HORIZONTAL_CONV { z -= 1; li -= bands; sum += coeff[z] * hfrom[li]; }


// JMCG BEGIN
// Vectorization
// Convolution function. input: short, output: int
#ifdef SIMD_WIDTH

void validate_conv_int_short_sse(REGION *ir, int le, int y, int lskip, int bands, ConvSequence *seq, int isz, int osz,
				 int *conv_sum, signed short *conv_out, int *coeff, Conv *conv, INTMASK *mask, int rounding) {
  signed short *vfrom;
  int *vto;
  int *hfrom;
  signed short *hto;
  int x,z, li;

  vfrom = (signed short *) IM_REGION_ADDR( ir, le, y );
  vto = conv_sum;

  for( x = 0; x < isz; x++ ) {
    int sum;
    z = conv->size;
    li = lskip * z;
    sum = 0;

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */
    if( (z) ) {
      int duff_count = ((z) + 15) / 16;

      switch( (z) % 16 ) {
      case 0:  do {   z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 15:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 14:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 13:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 12:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 11:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 10:      z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 9:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 8:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 7:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 6:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 5:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 4:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 3:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 2:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	case 1:       z -= 1; li -= lskip; sum += coeff[z] * vfrom[li];
	} while( --duff_count > 0 );
      }
    }
    sum = ((sum + rounding) / mask->scale) + mask->offset;
    vto[x] = sum;
    vfrom += 1;
  }

  /* Convolve sums to output.
   */
  hfrom = conv_sum;
  hto = conv_out;
  for( x = 0; x < osz; x++ ) {
    int sum;
    z = conv->size;
    li = bands * z;
    sum = 0;

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */
    if( (z) ) {
      int duff_count = ((z) + 15) / 16;
      switch( (z) % 16 ) {
      case 0:  do {   z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 15:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 14:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 13:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 12:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 11:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 10:      z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 9:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 8:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 7:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 6:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 5:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 4:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 3:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 2:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	case 1:       z -= 1; li -= bands; sum += coeff[z] * hfrom[li];
	} while( --duff_count > 0 );
      }
    }

    sum = ((sum + rounding) / mask->scale) + mask->offset;

    // IM_CLIP_SHORT
    if( sum < SHRT_MIN ) {
      seq->underflow++;
      sum = SHRT_MIN;
    }
    else if( sum > SHRT_MAX ) {
      seq->overflow++;
      sum = SHRT_MAX;
    }

    hto[x] = sum;
    hfrom += 1;
  }
}

#ifndef PARSEC_USE_NEON

static void inline conv_int_short_sse(REGION *ir, int le, int y, int lskip, int bands, ConvSequence *seq, int isz, int osz, int *coeff, Conv *conv, INTMASK *mask, int rounding, REGION *or) {
  int x,z, li;
  signed short *vfrom;
  int *vto;
  int *hfrom;
  signed short *hto;

#ifdef DEBUG_SIMD
  int validate_conv_sum[isz];
  signed short validate_conv_out[osz];
  validate_conv_int_short_sse(ir, le, y, lskip, bands, seq, isz, osz, &validate_conv_sum[0],&validate_conv_out[0], coeff, conv, mask, rounding);
#endif

  vfrom = (signed short *) IM_REGION_ADDR( ir, le, y );
  vto = (int *) seq->sum;

  for( x = 0; x < isz; x++ ) {
    __m128i _sum;
    _sum = _mm_setzero_si128();
    z = conv->size;
    li = lskip * z;
    int lskip_simd = lskip * 4;

    // JMCG Not using Duff's device at this point for Vectorization
    // Not sure if it is still usefull with automatic unroll from compiler
    // PRINT z, TRY MANUAL UNROLL to close number to see if Duff makes sense

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */

    while  ((z - 4) >= 0)  {
      li -= lskip_simd;
      z -= 4;
      //      sum += coeff[z] * vfrom[li];
      _sum = _mm_add_epi32(_sum,
			   _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((__m128i *) &coeff[z])),
						      _mm_set_ps(vfrom[li+lskip+lskip+lskip],vfrom[li+lskip+lskip],vfrom[li+lskip],vfrom[li])
						     )
					  )
			  );
    }

    _sum = _mm_hadd_epi32(_sum,_sum);
    _sum = _mm_hadd_epi32(_sum,_sum);

    // conv leftovers
    int sum = 0;
    while  (z > 0)  {
      li -= lskip;
      z -= 1;
      sum += coeff[z] * vfrom[li];
    }

    sum = ((sum + rounding + _mm_cvtsi128_si32(_sum)) / mask->scale) + mask->offset;
    vto[x] = sum;
    vfrom += 1;
  }

  /* Convolve sums to output.
   */
  hfrom = (int *) seq->sum;
  hto = (signed short *) IM_REGION_ADDR( or, le, y );
  for( x = 0; x < osz; x++ ) {
    __m128i _sum;
    _sum = _mm_setzero_si128();
    z = conv->size;
    li = bands * z;
    int band_simd = bands * 4;

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */

    while  ((z - 4) >= 0)  {
      li -= band_simd;
      z -= 4;
      // sum += coeff[z] * hfrom[li];
      _sum = _mm_add_epi32(_sum,
			   _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_loadu_si128((__m128i *) &coeff[z])),
						      _mm_set_ps(hfrom[li+bands+bands+bands],hfrom[li+bands+bands],hfrom[li+bands],hfrom[li])
						      )
					   )
			   );
    }

    _sum = _mm_hadd_epi32(_sum,_sum);
    _sum = _mm_hadd_epi32(_sum,_sum);

    // conv leftovers
    int sum = 0;
    while  (z > 0)  {
      li -= bands;
      z -= 1;
      sum += coeff[z] * hfrom[li];
    }

    sum = ((sum + rounding + _mm_cvtsi128_si32(_sum)) / mask->scale) + mask->offset;

    // IM_CLIP_SHORT
    if( sum < SHRT_MIN ) {
      seq->underflow++;
      sum = SHRT_MIN;
    }
    else if( sum > SHRT_MAX ) {
      seq->overflow++;
      sum = SHRT_MAX;
    }

    hto[x] = sum;
    hfrom += 1;
  }


#ifdef DEBUG_SIMD
  for(x = 0 ; x < isz; x++ ) {
    diff = abs(validate_conv_sum[x] - vto[x]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff Conv Sum = %f\n",max_diff);
      fflush(stdout);
    }
  }

  for(x = 0 ; x < osz; x++ ) {
    diff = abs(validate_conv_out[x] - hto[x]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff Conv Out = %f\n",max_diff);
      fflush(stdout);
    }
  }
#endif // DEBUG_SIMD
}

#else // PARSEC_USE_NEON

static void inline conv_int_short_neon(REGION *ir, int le, int y, int lskip, int bands, ConvSequence *seq, int isz, int osz, int *coeff, Conv *conv, INTMASK *mask, int rounding, REGION *or) {
  int x,z, li;
  signed short *vfrom;
  int *vto;
  int *hfrom;
  signed short *hto;

#ifdef DEBUG_SIMD
  int validate_conv_sum[isz];
  signed short validate_conv_out[osz];
  validate_conv_int_short_sse(ir, le, y, lskip, bands, seq, isz, osz, &validate_conv_sum[0],&validate_conv_out[0], coeff, conv, mask, rounding);
#endif

  vfrom = (signed short *) IM_REGION_ADDR( ir, le, y );
  vto = (int *) seq->sum;

  for( x = 0; x < isz; x++ ) {
    int32x4_t _sum;
    _sum = vdupq_n_s32(0);
    z = conv->size;
    li = lskip * z;
    int lskip_simd = lskip * 4;

    // JMCG Not using Duff's device at this point for Vectorization
    // Not sure if it is still usefull with automatic unroll from compiler
    // PRINT z, TRY MANUAL UNROLL to close number to see if Duff makes sense

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */

    while  ((z - 4) >= 0)  {
      li -= lskip_simd;
      z -= 4;
      //      sum += coeff[z] * vfrom[li];
      _sum = vaddq_s32(_sum,
		       vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(&coeff[z])),
					       _MM_SETM(vfrom[li+lskip+lskip+lskip],vfrom[li+lskip+lskip],vfrom[li+lskip],vfrom[li])
					       )
				     )
		       );
    }

    int32x2_t _tempsum = vpadd_s32(vget_low_s32(_sum),vget_high_s32(_sum));
    _tempsum = vpadd_s32(_tempsum,_tempsum);
    _sum = vcombine_s32(_tempsum,_tempsum);

    // conv leftovers
    int sum = 0;
    while  (z > 0)  {
      li -= lskip;
      z -= 1;
      sum += coeff[z] * vfrom[li];
    }

    sum = ((sum + rounding + vgetq_lane_s32(_sum,0)) / mask->scale) + mask->offset;
    vto[x] = sum;
    vfrom += 1;
  }

  /* Convolve sums to output.
   */
  hfrom = (int *) seq->sum;
  hto = (signed short *) IM_REGION_ADDR( or, le, y );
  for( x = 0; x < osz; x++ ) {
    int32x4_t _sum;
    _sum = vdupq_n_s32(0);
    z = conv->size;
    li = bands * z;
    int band_simd = bands * 4;

    /* Duff's device. Do OPERation N times in a 16-way unrolled loop.
     */

    while  ((z - 4) >= 0)  {
      li -= band_simd;
      z -= 4;
      // sum += coeff[z] * hfrom[li];

      _sum = vaddq_s32(_sum,
		       vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(&coeff[z])),
					       _MM_SETM(hfrom[li+bands+bands+bands],hfrom[li+bands+bands],hfrom[li+bands],hfrom[li])
					       )
				     )
		       );
    }

    int32x2_t _tempsum = vpadd_s32(vget_low_s32(_sum),vget_high_s32(_sum));
    _tempsum = vpadd_s32(_tempsum,_tempsum);
    _sum = vcombine_s32(_tempsum,_tempsum);

    // conv leftovers
    int sum = 0;
    while  (z > 0)  {
      li -= bands;
      z -= 1;
      sum += coeff[z] * hfrom[li];
    }

    sum = ((sum + rounding + vgetq_lane_s32(_sum,0)) / mask->scale) + mask->offset;

    // IM_CLIP_SHORT
    if( sum < SHRT_MIN ) {
      seq->underflow++;
      sum = SHRT_MIN;
    }
    else if( sum > SHRT_MAX ) {
      seq->overflow++;
      sum = SHRT_MAX;
    }

    hto[x] = sum;
    hfrom += 1;
  }


#ifdef DEBUG_SIMD
  for(x = 0 ; x < isz; x++ ) {
    diff = abs(validate_conv_sum[x] - vto[x]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff Conv Sum = %f\n",max_diff);
      fflush(stdout);
    }
  }

  for(x = 0 ; x < osz; x++ ) {
    diff = abs(validate_conv_out[x] - hto[x]);
    if (diff > max_diff) {
      max_diff = diff;
      printf("Maxdiff Conv Out = %f\n",max_diff);
      fflush(stdout);
    }
  }
#endif // DEBUG_SIMD
}


#endif // ifndef PARSEC_USE_NEON

#endif // SIMD_WIDTH
/* JMCG END */

/* INT and FLOAT inner loops.
 */
#define CONV_INT( TYPE, IM_CLIP ) { \
	TYPE *vfrom; \
	int *vto; \
	int *hfrom; \
	TYPE *hto; \
 	\
	/* Convolve to sum array. We convolve the full width of \
	 * this input line. \
	 */ \
	vfrom = (TYPE *) IM_REGION_ADDR( ir, le, y ); \
	vto = (int *) seq->sum; \
	for( x = 0; x < isz; x++ ) {   \
		int sum; \
		 \
		z = conv->size;  \
		li = lskip * z; \
		sum = 0; \
 		\
		IM_UNROLL( z, VERTICAL_CONV ); \
 		\
		sum = ((sum + rounding) / mask->scale) + mask->offset; \
		\
		vto[x] = sum;   \
		vfrom += 1; \
	}  \
 	\
	/* Convolve sums to output. \
	 */ \
	hfrom = (int *) seq->sum; \
	hto = (TYPE *) IM_REGION_ADDR( or, le, y );  \
	for( x = 0; x < osz; x++ ) { \
		int sum; \
		 \
		z = conv->size;  \
		li = bands * z; \
		sum = 0; \
 		\
		IM_UNROLL( z, HORIZONTAL_CONV ); \
 		\
		sum = ((sum + rounding) / mask->scale) + mask->offset; \
 		\
		IM_CLIP; \
 		\
		hto[x] = sum;   \
		hfrom += 1; \
	} \
}

#define CONV_FLOAT( TYPE ) { \
	TYPE *vfrom; \
	double *vto; \
	double *hfrom; \
	TYPE *hto; \
 	\
	/* Convolve to sum array. We convolve the full width of \
	 * this input line. \
	 */ \
	vfrom = (TYPE *) IM_REGION_ADDR( ir, le, y ); \
	vto = (double *) seq->sum; \
	for( x = 0; x < isz; x++ ) {   \
		double sum; \
		 \
		z = conv->size;  \
		li = lskip * z; \
		sum = 0; \
 		\
		IM_UNROLL( z, VERTICAL_CONV ); \
 		\
		vto[x] = sum;   \
		vfrom += 1; \
	}  \
 	\
	/* Convolve sums to output. \
	 */ \
	hfrom = (double *) seq->sum; \
	hto = (TYPE *) IM_REGION_ADDR( or, le, y );  \
	for( x = 0; x < osz; x++ ) { \
		double sum; \
		 \
		z = conv->size;  \
		li = bands * z; \
		sum = 0; \
 		\
		IM_UNROLL( z, HORIZONTAL_CONV ); \
 		\
		sum = (sum / conv->scale) + mask->offset; \
 		\
		hto[x] = sum;   \
		hfrom += 1; \
	} \
}

/* Convolve!
 */
static int
conv_gen( REGION *or, void *vseq, void *a, void *b )
{
	ConvSequence *seq = (ConvSequence *) vseq;
	IMAGE *in = (IMAGE *) a;
	Conv *conv = (Conv *) b;
	REGION *ir = seq->ir;
	INTMASK *mask = conv->mask;

	/* You might think this should be (scale+1)/2, but then we'd be adding
	 * one for scale == 1.
	 */
	int rounding = mask->scale / 2;

	int bands = in->Bands;
	int *coeff = conv->mask->coeff;

	Rect *r = &or->valid;
	int le = r->left;
	int to = r->top;
	int bo = IM_RECT_BOTTOM(r);
	int osz = IM_REGION_N_ELEMENTS( or );

	Rect s;
	int lskip;
	int isz;
	int x, y, z, li;

	/* Prepare the section of the input image we need. A little larger
	 * than the section of the output image we are producing.
	 */
	s = *r;
	s.width += conv->size - 1;
	s.height += conv->size - 1;
	if( im_prepare( ir, &s ) )
		return( -1 );
	lskip = IM_REGION_LSKIP( ir ) / IM_IMAGE_SIZEOF_ELEMENT( in );
	isz = IM_REGION_N_ELEMENTS( ir );

	for( y = to; y < bo; y++ ) {
		switch( in->BandFmt ) {
		case IM_BANDFMT_UCHAR:
			CONV_INT( unsigned char, IM_CLIP_UCHAR( sum, seq ) );
			break;
		case IM_BANDFMT_CHAR:
			CONV_INT( signed char, IM_CLIP_CHAR( sum, seq ) );
			break;
		case IM_BANDFMT_USHORT:
			CONV_INT( unsigned short, IM_CLIP_USHORT( sum, seq ) );
			break;
		case IM_BANDFMT_SHORT:
		  // JMCG Vectorization point, most of the time spent on this call
#ifdef SIMD_WIDTH
#ifndef PARSEC_USE_NEON
		  // SSE OR AVX, there is no AVX version for the convolution
		  conv_int_short_sse(ir, le, y, lskip, bands, seq, isz, osz, coeff, conv, mask, rounding, or);
#else
		  // NEON
		  conv_int_short_neon(ir, le, y, lskip, bands, seq, isz, osz, coeff, conv, mask, rounding, or);
#endif
#else
		  CONV_INT( signed short, IM_CLIP_SHORT( sum, seq ) );
#endif
		  break;
		case IM_BANDFMT_UINT:
			CONV_INT( unsigned int, IM_CLIP_NONE( sum, seq ) );
			break;
		case IM_BANDFMT_INT:
			CONV_INT( signed int, IM_CLIP_NONE( sum, seq ) );
			break;
		case IM_BANDFMT_FLOAT:
			CONV_FLOAT( float );
			break;
		case IM_BANDFMT_DOUBLE:
			CONV_FLOAT( double );
			break;

		default:
			assert( 0 );
		}
	}

	return( 0 );
}

int
im_convsep_raw( IMAGE *in, IMAGE *out, INTMASK *mask )
{
	Conv *conv;

	/* Check parameters.
	 */
	if( im_piocheck( in, out ) ||
		im_check_uncoded( "im_convsep", in ) ||
		im_check_noncomplex( "im_convsep", in ) ||
		im_check_imask( "im_convsep", mask ) )
		return( -1 );
	if( mask->xsize != 1 && mask->ysize != 1 ) {
                im_error( "im_convsep",
			"%s", _( "expect 1xN or Nx1 input mask" ) );
                return( -1 );
	}
	if( mask->scale == 0 ) {
		im_error( "im_convsep", "%s", "mask scale must be non-zero" );
		return( -1 );
	}
	if( !(conv = conv_new( in, out, mask )) )
		return( -1 );

	/* Prepare output. Consider a 7x7 mask and a 7x7 image --- the output
	 * would be 1x1.
	 */
	if( im_cp_desc( out, in ) )
		return( -1 );
	out->Xsize -= conv->size - 1;
	out->Ysize -= conv->size - 1;
	if( out->Xsize <= 0 || out->Ysize <= 0 ) {
		im_error( "im_convsep", "%s", _( "image too small for mask" ) );
		return( -1 );
	}

	/* SMALLTILE seems the fastest in benchmarks.
	 */
	if( im_demand_hint( out, IM_SMALLTILE, in, NULL ) ||
		im_generate( out, conv_start, conv_gen, conv_stop, in, conv ) )
		return( -1 );

	out->Xoffset = -mask->xsize / 2;
	out->Yoffset = -mask->ysize / 2;

	return( 0 );
}


/**
 * im_convsep:
 * @in: input image
 * @out: output image
 * @mask: convolution mask
 *
 * Perform a separable convolution of @in with @mask using integer arithmetic.
 *
 * The mask must be 1xn or nx1 elements.
 * The output image
 * always has the same #VipsBandFmt as the input image. Non-complex images
 * only.
 *
 * The image is convolved twice: once with @mask and then again with @mask
 * rotated by 90 degrees. This is much faster for certain types of mask
 * (gaussian blur, for example) than doing a full 2D convolution.
 *
 * Each output pixel is
 * calculated as sigma[i]{pixel[i] * mask[i]} / scale + offset, where scale
 * and offset are part of @mask. For integer @in, the division by scale
 * includes round-to-nearest.
 *
 * See also: im_convsep_f(), im_conv(), im_create_imaskv().
 *
 * Returns: 0 on success, -1 on error
 */
int
im_convsep( IMAGE *in, IMAGE *out, INTMASK *mask )
{
	IMAGE *t1 = im_open_local( out, "im_convsep intermediate", "p" );
	int size = mask->xsize * mask->ysize;

	if( !t1 ||
		im_embed( in, t1, 1, size / 2, size / 2,
			in->Xsize + size - 1,
			in->Ysize + size - 1 ) ||
		im_convsep_raw( t1, out, mask ) )
		return( -1 );

	out->Xoffset = 0;
	out->Yoffset = 0;

	return( 0 );
}
