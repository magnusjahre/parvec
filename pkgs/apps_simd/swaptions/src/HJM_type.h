// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

#ifndef __TYPE__
#define __TYPE__
//#define DEBUG

#if defined(BASELINE) && defined(ENABLE_SSE4)
#error BASELINE and ENABLE_SSE4 are mutually exclusive
#endif

/* JMCG BEGIN */

#define FTYPE float
//#define FTYPE double
//#define DFTYPE
// JMCG
#include "simd_defines.h"

/* END JMCG */

#define BLOCK_SIZE 64 // Blocking to allow better caching /* JMCG IF SSE or AVX is enabled make sure this is divisible by 4 / 8 respectively */

#define RANDSEEDVAL 100
#define DEFAULT_NUM_TRIALS  102400

typedef struct
{
  int Id;
  FTYPE dSimSwaptionMeanPrice;
  FTYPE dSimSwaptionStdError;
  FTYPE dStrike;
  FTYPE dCompounding;
  FTYPE dMaturity;
  FTYPE dTenor;
  FTYPE dPaymentInterval;
  int iN;
  FTYPE dYears;
  int iFactors;
  FTYPE *pdYield;
  FTYPE **ppdFactors;
} parm;



#endif //__TYPE__
