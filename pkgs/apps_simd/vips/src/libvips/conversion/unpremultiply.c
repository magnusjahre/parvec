/* unpremultiply alpha
 *
 * Author: John Cupitt
 * Written on: 7/5/15
 *
 */

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
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301  USA

 */

/*

    These files are distributed with VIPS - http://www.vips.ecs.soton.ac.uk

 */

/*
#define VIPS_DEBUG
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/
#include <vips/intl.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include <vips/vips.h>
#include <vips/internal.h>
#include <vips/debug.h>

#include "pconversion.h"

typedef struct _VipsUnpremultiply {
	VipsConversion parent_instance;

	VipsImage *in;

	double max_alpha;

} VipsUnpremultiply;

typedef VipsConversionClass VipsUnpremultiplyClass;

G_DEFINE_TYPE( VipsUnpremultiply, vips_unpremultiply, VIPS_TYPE_CONVERSION );

/* Unpremultiply a greyscale (two band) image.
 */
#define UNPRE_MANY( IN, OUT ) { \
	IN * restrict p = (IN *) in; \
	OUT * restrict q = (OUT *) out; \
	\
	for( x = 0; x < width; x++ ) { \
		IN alpha = p[bands - 1]; \
		int clip_alpha = VIPS_CLIP( 0, alpha, max_alpha ); \
		double nalpha = (double) clip_alpha / max_alpha; \
		\
		if( clip_alpha == 0 ) \
			for( i = 0; i < bands - 1; i++ ) \
				q[i] = 0; \
		else \
			for( i = 0; i < bands - 1; i++ ) \
				q[i] = p[i] / nalpha; \
		q[i] = clip_alpha; \
		\
		p += bands; \
		q += bands; \
	} \
}

/* Unpremultiply an RGB (four band) image.
 */
#define UNPRE_RGBA( IN, OUT ) { \
	IN * restrict p = (IN *) in; \
	OUT * restrict q = (OUT *) out; \
	\
	for( x = 0; x < width; x++ ) { \
		IN alpha = p[3]; \
		int clip_alpha = VIPS_CLIP( 0, alpha, max_alpha ); \
		double nalpha = (double) clip_alpha / max_alpha; \
		\
		if( clip_alpha == 0 ) { \
			q[0] = 0; \
			q[1] = 0; \
			q[2] = 0; \
		} \
		else { \
			q[0] = p[0] / nalpha; \
			q[1] = p[1] / nalpha; \
			q[2] = p[2] / nalpha; \
		} \
		q[3] = clip_alpha; \
		\
		p += 4; \
		q += 4; \
	} \
}

#define UNPRE( IN, OUT ) { \
	if( bands == 4 ) { \
		UNPRE_RGBA( IN, OUT ); \
	} \
	else { \
		UNPRE_MANY( IN, OUT ); \
	} \
}

static int
vips_unpremultiply_gen( VipsRegion *or, void *vseq, void *a, void *b,
	gboolean *stop )
{
	VipsUnpremultiply *unpremultiply = (VipsUnpremultiply *) b;
	VipsRegion *ir = (VipsRegion *) vseq;
	VipsImage *im = ir->im;
	VipsRect *r = &or->valid;
	int width = r->width;
	int bands = im->Bands; 
	double max_alpha = unpremultiply->max_alpha;

	int x, y, i;

	if( vips_region_prepare( ir, r ) )
		return( -1 );

	for( y = 0; y < r->height; y++ ) {
		VipsPel *in = VIPS_REGION_ADDR( ir, r->left, r->top + y ); 
		VipsPel *out = VIPS_REGION_ADDR( or, r->left, r->top + y ); 

		switch( im->BandFmt ) { 
		case VIPS_FORMAT_UCHAR: 
			UNPRE( unsigned char, float ); 
			break; 

		case VIPS_FORMAT_CHAR: 
			UNPRE( signed char, float ); 
			break; 

		case VIPS_FORMAT_USHORT: 
			UNPRE( unsigned short, float ); 
			break; 

		case VIPS_FORMAT_SHORT: 
			UNPRE( signed short, float ); 
			break; 

		case VIPS_FORMAT_UINT: 
			UNPRE( unsigned int, float ); 
			break; 

		case VIPS_FORMAT_INT: 
			UNPRE( signed int, float ); 
			break; 

		case VIPS_FORMAT_FLOAT: 
			UNPRE( float, float ); 
			break; 

		case VIPS_FORMAT_DOUBLE: 
			UNPRE( double, double ); 
			break; 

		case VIPS_FORMAT_COMPLEX: 
		case VIPS_FORMAT_DPCOMPLEX: 
		default: 
			g_assert_not_reached(); 
		} 
	}

	return( 0 );
}


static int
vips_unpremultiply_build( VipsObject *object )
{
	VipsObjectClass *class = VIPS_OBJECT_GET_CLASS( object );
	VipsConversion *conversion = VIPS_CONVERSION( object );
	VipsUnpremultiply *unpremultiply = (VipsUnpremultiply *) object;
	VipsImage **t = (VipsImage **) vips_object_local_array( object, 1 );

	VipsImage *in;

	if( VIPS_OBJECT_CLASS( vips_unpremultiply_parent_class )->
		build( object ) )
		return( -1 );

	in = unpremultiply->in; 

	if( vips_image_decode( in, &t[0] ) )
		return( -1 );
	in = t[0]; 

	/* Trivial case: fall back to copy().
	 */
	if( in->Bands == 1 )
		return( vips_image_write( in, conversion->out ) );

	if( vips_check_noncomplex( class->nickname, in ) )
		return( -1 );

	if( vips_image_pipelinev( conversion->out, 
		VIPS_DEMAND_STYLE_THINSTRIP, in, NULL ) )
		return( -1 );

	if( in->BandFmt == VIPS_FORMAT_DOUBLE )
		conversion->out->BandFmt = VIPS_FORMAT_DOUBLE;
	else
		conversion->out->BandFmt = VIPS_FORMAT_FLOAT;

	if( vips_image_generate( conversion->out,
		vips_start_one, vips_unpremultiply_gen, vips_stop_one, 
		in, unpremultiply ) )
		return( -1 );

	return( 0 );
}

static void
vips_unpremultiply_class_init( VipsUnpremultiplyClass *class )
{
	GObjectClass *gobject_class = G_OBJECT_CLASS( class );
	VipsObjectClass *vobject_class = VIPS_OBJECT_CLASS( class );
	VipsOperationClass *operation_class = VIPS_OPERATION_CLASS( class );

	VIPS_DEBUG_MSG( "vips_unpremultiply_class_init\n" );

	gobject_class->set_property = vips_object_set_property;
	gobject_class->get_property = vips_object_get_property;

	vobject_class->nickname = "unpremultiply";
	vobject_class->description = _( "unpremultiply image alpha" );
	vobject_class->build = vips_unpremultiply_build;

	operation_class->flags = VIPS_OPERATION_SEQUENTIAL_UNBUFFERED;

	VIPS_ARG_IMAGE( class, "in", 1, 
		_( "Input" ), 
		_( "Input image" ),
		VIPS_ARGUMENT_REQUIRED_INPUT,
		G_STRUCT_OFFSET( VipsUnpremultiply, in ) );

	VIPS_ARG_DOUBLE( class, "max_alpha", 115, 
		_( "Maximum alpha" ), 
		_( "Maximum value of alpha channel" ),
		VIPS_ARGUMENT_OPTIONAL_INPUT,
		G_STRUCT_OFFSET( VipsUnpremultiply, max_alpha ),
		0, 100000000, 255 );

}

static void
vips_unpremultiply_init( VipsUnpremultiply *unpremultiply )
{
	unpremultiply->max_alpha = 255.0;
}

/**
 * vips_unpremultiply:
 * @in: input image
 * @out: output image
 * @...: %NULL-terminated list of optional named arguments
 *
 * Optional arguments:
 *
 * @max_alpha: %gdouble, maximum value for alpha
 *
 * Unpremultiplies any alpha channel. 
 * The final band is taken to be the alpha
 * and the bands are transformed as:
 *
 * |[
 *   alpha = (int) clip( 0, in[in.bands - 1], @max_alpha ); 
 *   norm = (double) alpha / @max_alpha; 
 *   if( alpha == 0 )
 *   	out = [0, ..., 0, alpha];
 *   else
 *   	out = [in[0] / norm, ..., in[in.bands - 1] / norm, alpha];
 * ]|
 *
 * So for an N-band image, the first N - 1 bands are divided by the clipped 
 * and normalised final band, the final band is clipped. 
 * If there is only a single band, 
 * the image is passed through unaltered.
 *
 * The result is
 * #VIPS_FORMAT_FLOAT unless the input format is #VIPS_FORMAT_DOUBLE, in which
 * case the output is double as well.
 *
 * @max_alpha has the default value 255. You will need to set this to 65535
 * for images with a 16-bit alpha, or perhaps 1.0 for images with a float
 * alpha. 
 *
 * Non-complex images only.
 *
 * See also: vips_premultiply(), vips_flatten().
 *
 * Returns: 0 on success, -1 on error
 */
int
vips_unpremultiply( VipsImage *in, VipsImage **out, ... )
{
	va_list ap;
	int result;

	va_start( ap, out );
	result = vips_call_split( "unpremultiply", ap, in, out );
	va_end( ap );

	return( result );
}
