// CumNormalInv.c
// Author: Mark Broadie

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "HJM_type.h"

// JMCG
#ifdef PARSEC_USE_SSE
#include "sse_mathfun.h"
#endif

#ifdef PARSEC_USE_AVX
#include "avx_mathfun.h"
#endif

#ifdef PARSEC_USE_NEON
#include "neon_mathfun.h"
#endif

#include <iostream>
#include <string.h>

using namespace std;

FTYPE CumNormalInv( FTYPE u );

/* JMCG */

//#define DEBUG_SIMD

/**********************************************************************/

#ifdef SIMD_WIDTH

static FTYPE a_orig[4] = {
  2.50662823884,
  -18.61500062529,
  41.39119773534,
    -25.44106049637
};

static FTYPE b_orig[4] = {
  -8.47351093090,
  23.08336743743,
  -21.06224101826,
    3.13082909833
};

static FTYPE c_orig[9] = {
  0.3374754822726147,
  0.9761690190917186,
  0.1607979714918209,
  0.0276438810333863,
  0.0038405729373609,
  0.0003951896511919,
  0.0000321767881768,
  0.0000002888167364,
    0.0000003960315187
};


/* JMCG Defines for SIMD versions */
#if (SIMD_WIDTH == 2)
static _MM_ALIGN FTYPE a[4][2] = { { 2.50662823884, 2.50662823884 },
				   { -18.61500062529, -18.61500062529 },
				   { 41.39119773534, 41.39119773534 },
				   { -25.44106049637, -25.44106049637 }
};

static _MM_ALIGN FTYPE b[4][2] = { { -8.47351093090, -8.47351093090 },
				   { 23.08336743743, 23.08336743743 },
				   { -21.06224101826, -21.06224101826 },
				   { 3.13082909833, 3.13082909833 }
};


static _MM_ALIGN FTYPE c[9][2] = { { 0.3374754822726147, 0.3374754822726147 },
				   { 0.9761690190917186, 0.9761690190917186 },
				   { 0.1607979714918209, 0.1607979714918209 },
				   { 0.0276438810333863, 0.0276438810333863 },
				   { 0.0038405729373609, 0.0038405729373609 },
				   { 0.0003951896511919, 0.0003951896511919 },
				   { 0.0000321767881768, 0.0000321767881768 },
				   { 0.0000002888167364, 0.0000002888167364 },
				   { 0.0000003960315187, 0.0000003960315187 }
};

#endif
#if (SIMD_WIDTH == 4)
static _MM_ALIGN FTYPE a[4][4] = { { 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884 },
                                   { -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529 },
                                   { 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534 },
                                   { -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637 }
};

static _MM_ALIGN FTYPE b[4][4] = { { -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090 },
                                   { 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743 },
                                   { -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826 },
                                   { 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833 }
};

static _MM_ALIGN FTYPE c[9][4] = { { 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147 },
                                   { 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186 },
                                   { 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209 },
                                   { 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863 },
                                   { 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609 },
                                   { 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919 },
                                   { 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768 },
                                   { 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364 },
                                   { 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187 }
                                 };

#endif
#if (SIMD_WIDTH == 8)
static _MM_ALIGN FTYPE a[4][8] = { { 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884, 2.50662823884 },
                                   { -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529, -18.61500062529 },
                                   { 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534, 41.39119773534 },
                                   { -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637, -25.44106049637 }
};



static _MM_ALIGN FTYPE b[4][8] = { { -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090, -8.47351093090 },
                                   { 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743, 23.08336743743 },
                                   { -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826, -21.06224101826 },
                                   { 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833, 3.13082909833 }
};



static _MM_ALIGN FTYPE c[9][8] = { { 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147, 0.3374754822726147 },
                                   { 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186, 0.9761690190917186 },
                                   { 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209, 0.1607979714918209 },
                                   { 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863, 0.0276438810333863 },
                                   { 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609, 0.0038405729373609 },
                                   { 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919, 0.0003951896511919 },
                                   { 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768, 0.0000321767881768 },
                                   { 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364, 0.0000002888167364 },
                                   { 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187, 0.0000003960315187 }
};
#endif

#else // ! SIMD_WIDTH
static FTYPE a_orig[4] = {
  2.50662823884,
    -18.61500062529,
    41.39119773534,
    -25.44106049637
};

static FTYPE b_orig[4] = {
  -8.47351093090,
    23.08336743743,
    -21.06224101826,
    3.13082909833
};

static FTYPE c_orig[9] = {
  0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187
};
#endif // SIMD_WIDTH

/* END JMCG */

/**********************************************************************/
#ifndef SIMD_WIDTH
FTYPE CumNormalInv( FTYPE u )
{
  // Returns the inverse of cumulative normal distribution function.
  // Reference: Moro, B., 1995, "The Full Monte," RISK (February), 57-58.

  FTYPE x, r;

#ifdef DEBUG_SIMD
  cout << "U " << u << endl;
#endif

  x = u - 0.5;

#ifdef DEBUG_SIMD
  cout << "X " << x << endl;
#endif

  if( fabs (x) < 0.42 )
  {
    r = x * x;
    r = x * ((( a_orig[3]*r + a_orig[2]) * r + a_orig[1]) * r + a_orig[0])/
          ((((b_orig[3] * r+ b_orig[2]) * r + b_orig[1]) * r + b_orig[0]) * r + 1.0);

#ifdef DEBUG_SIMD
      cout << "Output Firstif: " << r << endl;
#endif

    return (r);
  }

  r = u;
  if( x > 0.0 ) r = 1.0 - u;
  r = log(-log(r));
  r = c_orig[0] + r * (c_orig[1] + r *
       (c_orig[2] + r * (c_orig[3] + r *
       (c_orig[4] + r * (c_orig[5] + r * (c_orig[6] + r * (c_orig[7] + r*c_orig[8])))))));
  if( x < 0.0 ) {
    r = -r;
#ifdef DEBUG_SIMD
    cout << "Output Second if: " << r << endl;
#endif
  } else {
#ifdef DEBUG_SIMD
    cout << "Output Third if: " << r << endl;
#endif
  }


  return (r);

} // end of CumNormalInv

#else // SIMD_WIDTH
// Only for debug

//FTYPE CumNormalInv( FTYPE u ) { }

FTYPE CumNormalInv( FTYPE u )
{
  // Returns the inverse of cumulative normal distribution function.
  // Reference: Moro, B., 1995, "The Full Monte," RISK (February), 57-58.

  FTYPE x, r;

#ifdef DEBUG_SIMD
  cout << "U " << u << endl;
#endif

  x = u - 0.5;

#ifdef DEBUG_SIMD
  cout << "X " << x << endl;
#endif

  if( fabs (x) < 0.42 )
    {
      r = x * x;

#ifdef DEBUG_SIMD
      cout << "R: " << r << endl;
#endif

      r = x * ((( a_orig[3]*r + a_orig[2]) * r + a_orig[1]) * r + a_orig[0])/
	((((b_orig[3] * r+ b_orig[2]) * r + b_orig[1]) * r + b_orig[0]) * r + 1.0);

#ifdef DEBUG_SIMD
      cout << "Output Firstif: " << r << endl;
#endif
      return (r);
    }

  r = u;
  if( x > 0.0 ) r = 1.0 - u;
  r = log(-log(r));
  r = c_orig[0] + r * (c_orig[1] + r *
		       (c_orig[2] + r * (c_orig[3] + r *
					 (c_orig[4] + r * (c_orig[5] + r * (c_orig[6] + r * (c_orig[7] + r*c_orig[8])))))));

  if( x < 0.0 ) {
    r = -r;
#ifdef DEBUG_SIMD
    cout << "Output Second if: " << r << endl;
#endif
  } else {
#ifdef DEBUG_SIMD
    cout << "Output Third if: " << r << endl;
#endif
  }

  return (r);
} // end of CumNormalInv

/**********************************************************************/

/* JMCG */
/*
   Input: "u" | Pointer to SIMD_WIDTH (e.g.2, 4, 8) elements of FTYPE (i.e., float, double) type
   Output: Output from the function is stored on a preallocated memory location given by the FTYPE pointer "output"
 */

void CumNormalInv_simd( FTYPE* u, FTYPE* output )
{
  // R_xeturns the inverse of cumulative normal distribution function.
  // Reference: Moro, B., 1995, "The Full Monte," RISK (February), 57-58.

  bool skip_part_1, skip_part_2;
  _MM_TYPE answer, x, _xabs, _part1, _part2, _flag;
  _MM_TYPE temp = _MM_SET(0.5);

  _MM_ALIGN FTYPE output2[SIMD_WIDTH];
  skip_part_1 = 0;
  skip_part_2 = 0;

  //   x = u - 0.5;
  _part2 = _MM_LOAD(u);

#ifdef DEBUG_SIMD
  _MM_STORE(output2,_part2);
  cout << "Input SIMD " << endl;
  for (int i = 0; i< SIMD_WIDTH; i++) {
    cout << output2[i] << " " ;
  }
  cout << endl;
#endif

  x = _MM_SUB(_part2,temp);

#ifdef DEBUG_SIMD
  _MM_STORE(output2,x);
  cout << "X SIMD " << endl;
  for (int i = 0; i< SIMD_WIDTH; i++) {
    cout << output2[i] << " " ;
  }
  cout << endl;
#endif


  // If statement is no longer possible with SSE (as we load 2/4/8 values at the same time
  // We can have 3 scenarios.  Case (all x) < .42, case (all x) >= .42 and "other"

  //if( fabs (x) < 0.42 )
  _xabs = _MM_ABS(x);
  _flag = (_MM_TYPE)_MM_CMPLT(_xabs, _MM_SET(0.42));

  //mm_movemask_pd ps
  int x_most_sig_bits = _MM_MOVEMASK(_flag); /* high bits of each of our bools */

#ifdef DEBUG_SIMD
  printf("SigBits %d\n",x_most_sig_bits);
#endif

  if (x_most_sig_bits == 0) {
    skip_part_1 = 1;
  }

  if (x_most_sig_bits == _MM_MASK_TRUE) {
    skip_part_2 = 1;
    //    printf("Bits %d\n",x_most_sig_bits);
  }

  //{
  //  r = x * x;
  //  r = x * ((( a[3]*r + a[2]) * r + a[1]) * r + a[0])/
  //        ((((b[3] * r+ b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
  //  return (r);
  // }

  // We always run this unless all values are higher than .42

  if(skip_part_1 == 0) {
    _part1 = _MM_MUL(x,x);


#ifdef DEBUG_SIMD
    _MM_STORE(output2,_part1);
    cout << "X x X " << endl;
    for (int i = 0; i< SIMD_WIDTH; i++) {
      cout << output2[i] << " " ;
    }
    cout << endl;
#endif

    _part1 = _MM_MUL(x,_MM_DIV(_MM_ADD(_MM_MUL(_MM_ADD(_MM_MUL(_MM_ADD(_MM_MUL(_MM_LOAD(a[3]), _part1), _MM_LOAD(a[2])), _part1), _MM_LOAD(a[1])), _part1), _MM_LOAD(a[0])),
			       _MM_ADD(_MM_MUL(_MM_ADD(_MM_MUL(_MM_ADD(_MM_MUL(_MM_ADD(_MM_MUL(_MM_LOAD(b[3]), _part1), _MM_LOAD(b[2])), _part1), _MM_LOAD(b[1])), _part1), _MM_LOAD(b[0])),_part1), _MM_SET(1))));

#ifdef DEBUG_SIMD
    _MM_STORE(output2,_part1);
    cout << "part1 " << endl;
    for (int i = 0; i< SIMD_WIDTH; i++) {
      cout << output2[i] << " " ;
    }
    cout << endl;
#endif

  }

  // If statement is no longer possible with SSE, we run this part unless all values are lower than .42
  if(skip_part_2 == 0) {
    //  r = u; // _part2 already has this value (r)

    // if( x > 0.0 ) r = 1.0 - u;
    // Get signs of x
    _flag = (_MM_TYPE)_MM_CMPLT(x, _MM_SET(0));
    _part2 = _MM_OR(_MM_AND(_flag, _part2), _MM_ANDNOT(_flag, _MM_SUB(_MM_SET(1.0), _part2)));


    //  r = log(-log(r));
    _part2 = _MM_LOG(_MM_MUL(_MM_LOG(_part2),_MM_SET(-1)));


    //  r = c[0] + r * (c[1] + r *
    //       (c[2] + r * (c[3] + r *
    //       (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r*c[8]))))))); // JMCG Original code profile shows this takes 2.5% of execution time,
                                                                        // vectorized takes 5%, not sure why it takes longer, maybe the compiler cannot reorganize loads on the Vectorized code.
                                                                        // Manual reorganization of the function may lead to more speeups

    _part2 = _MM_ADD(_MM_LOAD(c[0]), _MM_MUL(_part2,
					     _MM_ADD(_MM_LOAD(c[1]), _MM_MUL(_part2,_MM_ADD(_MM_LOAD(c[2]), _MM_MUL(_part2,_MM_ADD(_MM_LOAD(c[3]), _MM_MUL(_part2,
																			   _MM_ADD(_MM_LOAD(c[4]), _MM_MUL(_part2,_MM_ADD(_MM_LOAD(c[5]),_MM_MUL(_part2,_MM_ADD(_MM_LOAD(c[6]),
																			   _MM_MUL(_part2, _MM_ADD(_MM_LOAD(c[7]), _MM_MUL(_part2,_MM_LOAD(c[8])))))))))))))))));


  }

    //  if( x < 0.0 ) r = -r;
  _flag = (_MM_TYPE)_MM_CMPLT(x, _MM_SET(0));
    _part2 = _MM_OR(_MM_AND(_flag, _MM_MUL(_MM_SET(-1.0), _part2)), _MM_ANDNOT(_flag, _part2));

#ifdef DEBUG_SIMD
    _MM_STORE(output2,_xabs);
    cout << "ABS X " << endl;
    for (int i = 0; i< SIMD_WIDTH; i++) {
      cout << output2[i] << " " ;
    }
    cout << endl;
#endif

    _flag = (_MM_TYPE)_MM_CMPLT(_xabs, _MM_SET(0.42));

#ifdef DEBUG_SIMD
    _MM_STORE(output2,_flag);
    cout << "FLAG " << endl;
    for (int i = 0; i< SIMD_WIDTH; i++) {
      cout << output2[i] << " " ;
    }
    cout << endl;
#endif

    // Compose answer (_part1 or _part2 based on ABS(x) < 0.42)
    answer = _MM_OR(_MM_AND(_flag, _part1), _MM_ANDNOT(_flag, _part2));

#ifdef DEBUG_SIMD
    _MM_STORE(output2,answer);
    cout << "Final Output SIMD " << endl;
    for (int i = 0; i< SIMD_WIDTH; i++) {
      cout << output2[i] << " " ;
    }
    cout << endl;
#endif


    _MM_STORE(output,answer);

    return;

  } // end of CumNormalInv

#endif

  /* END JMCG */

/**********************************************************************/

// end of CumNormalInv.c
