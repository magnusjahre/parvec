/* draw a histogram
 *
 * Copyright: 1990, N. Dessipris.
 *
 * Author: Nicos Dessipris.
 * Written on: 09/07/1990
 * Modified on : 12/03/1991
 * 20/6/95 JC
 *	- rules rationalised
 *	- im_lineprof removed
 *	- rewritten
 * 13/8/99 JC
 *	- rewritten again for partial, rules redone
 * 19/9/99 JC
 *	- oooops, broken for >1 band
 * 26/9/99 JC
 *	- oooops, graph float was wrong
 * 17/11/99 JC
 *	- oops, failed for all 0's histogram 
 * 14/12/05
 * 	- redone plot function in C, also use incheck() to cache calcs
 * 	- much, much faster!
 * 12/5/09
 *	- fix signed/unsigned warning
 * 24/3/10
 * 	- gtkdoc
 * 	- small cleanups
 * 	- oop, would fail for signed int histograms
 * 19/8/13
 * 	- wrap as a class, left a rewrite for now
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

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif /*HAVE_CONFIG_H*/
#include <vips/intl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vips/vips.h>

#include "phistogram.h"

static int
plotalise( IMAGE *in, IMAGE *out )
{
	if( im_check_uncoded( "im_histplot", in ) ||
		im_check_noncomplex( "im_histplot", in ) )
		return( -1 );

	if( vips_bandfmt_isuint( in->BandFmt ) ) {
		if( im_copy( in, out ) )
			return( -1 );
	}
	else if( vips_bandfmt_isint( in->BandFmt ) ) {
		double min;

		/* Move min up to 0. 
		 */
		if( im_min( in, &min ) ||
			im_lintra( 1.0, in, -min, out ) )
			return( -1 );
	}
	else {
		/* Float image: scale min--max to 0--any. Output square
		 * graph.
		 */
		DOUBLEMASK *stats;
		double min, max;
		int any;

		if( in->Xsize == 1 )
			any = in->Ysize;
		else
			any = in->Xsize;

		if( !(stats = im_stats( in )) )
			return( -1 );
		min = IM_MASK( stats, 0, 0 );
		max = IM_MASK( stats, 1, 0 );
		im_free_dmask( stats );

		if( im_lintra( any / (max - min), in, 
			-min * any / (max - min), out ) )
			return( -1 );
	}

	return( 0 );
}

#define VERT( TYPE ) { \
	TYPE *p1 = (TYPE *) p; \
	\
	for( x = le; x < ri; x++ ) { \
		for( z = 0; z < nb; z++ )  \
			q[z] = p1[z] < ((TYPE) x) ? 0 : 255; \
		\
		q += nb; \
	} \
}

/* Generate function.
 */
static int
make_vert_gen( REGION *or, void *seq, void *a, void *b )
{
	IMAGE *in = (IMAGE *) a;
	Rect *r = &or->valid;
	int le = r->left;
	int to = r->top;
	int ri = IM_RECT_RIGHT( r );
	int bo = IM_RECT_BOTTOM( r );
	int nb = in->Bands;

	int x, y, z;

	for( y = to; y < bo; y++ ) {
		VipsPel *q = IM_REGION_ADDR( or, le, y );
		VipsPel *p = IM_IMAGE_ADDR( in, 0, y );

		switch( in->BandFmt ) {
		case IM_BANDFMT_UCHAR: 	VERT( unsigned char ); break;
		case IM_BANDFMT_CHAR: 	VERT( signed char ); break; 
		case IM_BANDFMT_USHORT: VERT( unsigned short ); break; 
		case IM_BANDFMT_SHORT: 	VERT( signed short ); break; 
		case IM_BANDFMT_UINT: 	VERT( unsigned int ); break; 
		case IM_BANDFMT_INT: 	VERT( signed int );  break; 
		case IM_BANDFMT_FLOAT: 	VERT( float ); break; 
		case IM_BANDFMT_DOUBLE:	VERT( double ); break; 

		default:
			g_assert_not_reached(); 
		}
	}

	return( 0 );
}

#define HORZ( TYPE ) { \
	TYPE *p1 = (TYPE *) p; \
	\
	for( y = to; y < bo; y++ ) { \
		for( z = 0; z < nb; z++ )  \
			q[z] = p1[z] < ((TYPE) (ht - y)) ? 0 : 255; \
		\
		q += lsk; \
	} \
}

/* Generate function.
 */
static int
make_horz_gen( REGION *or, void *seq, void *a, void *b )
{
	IMAGE *in = (IMAGE *) a;
	Rect *r = &or->valid;
	int le = r->left;
	int to = r->top;
	int ri = IM_RECT_RIGHT( r );
	int bo = IM_RECT_BOTTOM( r );
	int nb = in->Bands;
	int lsk = IM_REGION_LSKIP( or );
	int ht = or->im->Ysize;

	int x, y, z;

	for( x = le; x < ri; x++ ) {
		VipsPel *q = IM_REGION_ADDR( or, x, to );
		VipsPel *p = IM_IMAGE_ADDR( in, x, 0 );

		switch( in->BandFmt ) {
		case IM_BANDFMT_UCHAR: 	HORZ( unsigned char ); break;
		case IM_BANDFMT_CHAR: 	HORZ( signed char ); break; 
		case IM_BANDFMT_USHORT: HORZ( unsigned short ); break; 
		case IM_BANDFMT_SHORT: 	HORZ( signed short ); break; 
		case IM_BANDFMT_UINT: 	HORZ( unsigned int ); break; 
		case IM_BANDFMT_INT: 	HORZ( signed int );  break; 
		case IM_BANDFMT_FLOAT: 	HORZ( float ); break; 
		case IM_BANDFMT_DOUBLE:	HORZ( double ); break; 

		default:
			g_assert_not_reached();
		}
	}

	return( 0 );
}

/* Plot image.
 */
static int
plot( IMAGE *in, IMAGE *out )
{
	double max;
	int tsize;
	int xsize;
	int ysize;

	if( im_incheck( in ) ||
		im_poutcheck( out ) )
		return( -1 );

	/* Find range we will plot.
	 */
	if( im_max( in, &max ) )
		return( -1 );
	g_assert( max >= 0 );
	if( in->BandFmt == IM_BANDFMT_UCHAR )
		tsize = 256;
	else
		tsize = VIPS_CEIL( max );

	/* Make sure we don't make a zero height image.
	 */
	if( tsize == 0 )
		tsize = 1;

	if( in->Xsize == 1 ) {
		/* Vertical graph.
		 */
		xsize = tsize;
		ysize = in->Ysize;
	}
	else {
		/* Horizontal graph.
		 */
		xsize = in->Xsize;
		ysize = tsize;
	}

	/* Set image.
	 */
	im_initdesc( out, xsize, ysize, in->Bands, 
		IM_BBITS_BYTE, IM_BANDFMT_UCHAR, 
		IM_CODING_NONE, IM_TYPE_HISTOGRAM, 1.0, 1.0, 0, 0 );

	/* Set hints - ANY is ok with us.
	 */
	if( im_demand_hint( out, IM_ANY, NULL ) )
		return( -1 );
	
	/* Generate image.
	 */
	if( in->Xsize == 1 ) {
		if( im_generate( out, NULL, make_vert_gen, NULL, in, NULL ) )
			return( -1 );
	}
	else {
		if( im_generate( out, NULL, make_horz_gen, NULL, in, NULL ) )
			return( -1 );
	}

	return( 0 );
}

int 
im_histplot( IMAGE *in, IMAGE *out )
{
	IMAGE *t1;

	if( im_check_hist( "im_histplot", in ) )
		return( -1 );

	if( !(t1 = im_open_local( out, "im_histplot:1", "p" )) ||
		plotalise( in, t1 ) ||
		plot( t1, out ) )
		return( -1 );
	
	return( 0 );
}

typedef struct _VipsHistPlot {
	VipsOperation parent_instance;

	VipsImage *in;
	VipsImage *out;
} VipsHistPlot;

typedef VipsOperationClass VipsHistPlotClass;

G_DEFINE_TYPE( VipsHistPlot, vips_hist_plot, VIPS_TYPE_OPERATION );

static int
vips_hist_plot_build( VipsObject *object )
{
	VipsHistPlot *plot = (VipsHistPlot *) object;

	g_object_set( plot, "out", vips_image_new(), NULL ); 

	if( VIPS_OBJECT_CLASS( vips_hist_plot_parent_class )->build( object ) )
		return( -1 );

	if( im_histplot( plot->in, plot->out ) )
		return( -1 ); 

	return( 0 );
}

static void
vips_hist_plot_class_init( VipsHistPlotClass *class )
{
	GObjectClass *gobject_class = G_OBJECT_CLASS( class );
	VipsObjectClass *object_class = (VipsObjectClass *) class;

	gobject_class->set_property = vips_object_set_property;
	gobject_class->get_property = vips_object_get_property;

	object_class->nickname = "hist_plot";
	object_class->description = _( "plot histogram" );
	object_class->build = vips_hist_plot_build;

	VIPS_ARG_IMAGE( class, "in", 1, 
		_( "Input" ), 
		_( "Input image" ),
		VIPS_ARGUMENT_REQUIRED_INPUT,
		G_STRUCT_OFFSET( VipsHistPlot, in ) );

	VIPS_ARG_IMAGE( class, "out", 2, 
		_( "Output" ), 
		_( "Output image" ),
		VIPS_ARGUMENT_REQUIRED_OUTPUT, 
		G_STRUCT_OFFSET( VipsHistPlot, out ) );
}

static void
vips_hist_plot_init( VipsHistPlot *hist_plot )
{
}

/**
 * vips_hist_plot:
 * @in: input image
 * @out: output image
 * @...: %NULL-terminated list of optional named arguments
 *
 * Plot a 1 by any or any by 1 image file as a max by any or 
 * any by max image using these rules:
 * 
 * <emphasis>unsigned char</emphasis> max is always 256 
 *
 * <emphasis>other unsigned integer types</emphasis> output 0 - maxium 
 * value of @in.
 *
 * <emphasis>signed int types</emphasis> min moved to 0, max moved to max + min.
 *
 * <emphasis>float types</emphasis> min moved to 0, max moved to any 
 * (square output)
 *
 * Returns: 0 on success, -1 on error
 */
int 
vips_hist_plot( VipsImage *in, VipsImage **out, ... )
{
	va_list ap;
	int result;

	va_start( ap, out );
	result = vips_call_split( "hist_plot", ap, in, out );
	va_end( ap );

	return( result );
}


