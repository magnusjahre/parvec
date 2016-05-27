// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice
// Hall, John C. Hull,

// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef WIN32

// JMCG Need to define both
#define fptype float

/* JMCG BEGIN */
#ifdef PARSEC_USE_SSE
#include "sse_mathfun.h"
#include "simd_defines.h"
#endif

#ifdef PARSEC_USE_AVX
#include "avx_mathfun.h"
#include "simd_defines.h"
#endif

#ifdef PARSEC_USE_NEON
#include "neon_mathfun.h"
#include "simd_defines.h"
#endif
/* JMCG END */

#else
#include <xmmintrin.h>
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif


// #define DEBUG_SIMD
#ifdef DEBUG_SIMD
#define ERR_CHK
#endif

// We need this to compile the scalar version
#ifndef SIMD_WIDTH
#ifdef __GNUC__
#define _MM_ALIGN __attribute__((aligned (16)))
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif
#endif
// END JMCG

// Multi-threaded pthreads header
#ifdef ENABLE_THREADS
// Add the following line so that icc 9.0 is compatible with pthread lib.
#define __thread __threadp
MAIN_ENV
#undef __thread
#endif

// Multi-threaded OpenMP header
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_TBB
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"

using namespace std;
using namespace tbb;
#endif //ENABLE_TBB

// Multi-threaded header for Windows
#ifdef WIN32
#pragma warning(disable : 4305)
#pragma warning(disable : 4244)
#include <windows.h>
#endif

#ifdef __GNUC__
//#define _MM_ALIGN __attribute__((aligned (16))) // JMCG We don't use this
#define MUSTINLINE __attribute__((always_inline))
#else
#define MUSTINLINE __forceinline
#endif

#define NUM_RUNS 100

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

_MM_ALIGN OptionData* data; // JMCG
_MM_ALIGN fptype* prices; // JMCG
int numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;
int nThreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

/* JMCG */

#ifdef SIMD_WIDTH

MUSTINLINE _MM_TYPE CNDF_SIMD ( _MM_TYPE InputX ) {

  _MM_TYPE _x;
  _x = InputX;

  _MM_TYPE _k, _n, _accum, _candidate_answer, _flag,
    _A1 = _MM_SET(0.319381530),
    _A2 = _MM_SET(-0.356563782),
    _A3 = _MM_SET(1.781477937),
    _A4 = _MM_SET(-1.821255978),
    _A5 = _MM_SET(1.330274429),
    _INV_ROOT2PI = _MM_SET(0.39894228);

  //Get signs of _x
  _flag = (_MM_TYPE)_MM_CMPLT(_x, _MM_SET(0));

  //Get absolute value of x
  _x = _MM_ABS(_x);

  // k = 1.0 / (1.0 + 0.2316419 * x);
  _k = _MM_DIV(_MM_SET(1), _MM_ADD(_MM_SET(1), _MM_MUL(_MM_SET(0.2316419), _x)));

  _accum = _MM_ADD(_A4, _MM_MUL(_A5, _k));
  _accum = _MM_ADD(_A3, _MM_MUL(_accum, _k));
  _accum = _MM_ADD(_A2, _MM_MUL(_accum, _k));
  _accum = _MM_ADD(_A1, _MM_MUL(_accum, _k));
  _accum = _MM_MUL(_accum, _k);

  // n = expf(-0.5 * x * x);
  // n *= INV_ROOT2PI;
  _n = _MM_MUL(_MM_EXP(_MM_MUL(_MM_MUL(_MM_SET(-.5), _x), _x)), _INV_ROOT2PI);

  // candidate_answer = 1.0 - n * accum;
  _candidate_answer = _MM_SUB(_MM_SET(1), _MM_MUL(_n, _accum));
  // return (flag ? 1.0 - candidate_answer : candidate_answer);
  _candidate_answer = _MM_OR(_MM_ANDNOT(_flag, _candidate_answer),
                                _MM_AND(_flag, _MM_SUB(_MM_SET(1), _candidate_answer)));
  return _candidate_answer;
}


//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void BlkSchlsEqEuroNoDiv_SIMD (fptype * OptionPrice, int numOptions, fptype * sptprice,
			       fptype * strike, fptype * rate, fptype * volatility,
			       fptype * time, int * otype, float timet)
{

  _MM_TYPE _d1, _d2, _c, _p, _Nd1, _Nd2, _expval, _answer, _tmp1, _T, _sigma, _K, _r, _S0;

  //Loads
  _T      = _MM_LOADU(time);
  _sigma  = _MM_LOADU(volatility);
  _K      = _MM_LOADU(strike);
  _r      = _MM_LOADU(rate);
  _S0     = _MM_LOADU(sptprice);

  // d1 = logf(S0/K)
  _d1     = _MM_DIV(_S0, _K);
  _d1     = _MM_LOG(_d1);

  // d1 = logf(S0/K) + (r + 0.5*sigma*sigma)*T;
  _tmp1 = _MM_MUL(_sigma, _sigma);           // sigma*sigma
  _tmp1 = _MM_MUL(_tmp1, _MM_SET(.5)); // 0.5*sigma*sigma
  _tmp1 = _MM_ADD(_tmp1, _r);                // r + 0.5*sigma*sigma
  _tmp1 = _MM_MUL(_tmp1, _T);                // (r + 0.5*sigma*sigma)*T
  _d1   = _MM_ADD(_d1, _tmp1);               // logf(S0/K) + (r + 0.5*sigma*sigma)*T

  _MM_TYPE _sqrt_T = _MM_SQRT(_T);
  // d1 /= (sigma * sqrt(T));
  _d1   = _MM_DIV(_d1, _sigma);   // d1 /= sigma
  _d1   = _MM_DIV(_d1, _sqrt_T);     // d1 /= (sigma * sqrt(T))

  // d2 = d1 - sigma * sqrt(T);
  _d2   = _MM_SUB(_d1, _MM_MUL(_sigma, _sqrt_T));

  _Nd1 = CNDF_SIMD(_d1);
  _Nd2 = CNDF_SIMD(_d2);

  // expval = exp(-r*T)
  _expval = _MM_MUL(_T, _r);
  //Negate value of r by reversing the sign bit
  //  _MM_TYPE _absmask = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));
  // _expval = _mm_xor_ps(_absmask, _expval);
  _expval = _MM_NEG(_expval);

  _expval = _MM_EXP(_expval);

  // c = S0 * Nd1 - K * expval * Nd2;
  _c = _MM_SUB(_MM_MUL(_S0, _Nd1), _MM_MUL(_MM_MUL(_K, _expval), _Nd2));

  // p = K * expval * (1.0 - Nd2) - S0 * (1.0 - Nd1);
  _p = _MM_SUB(_MM_MUL(_K, _MM_MUL(_expval, _MM_SUB(_MM_SET(1), _Nd2))), // K * expval * (1.0 - Nd2)
                  _MM_MUL(_S0, _MM_SUB(_MM_SET(1), _Nd1))); // S0 * (1.0 - Nd1)

  //  _tmp1 = (_MM_TYPE)_MM_CMPEQ(_MM_LOADU((float*)otype), _MM_SETZERO()); // otype ?
  // This looks weird but our ARM evaluation system seems to be running in
  // Runfast mode. In this mode Subnormal numbers are being flushed to zero (that is, the 0x0...1 stored in otype)
  // Casting everything to integer and using integer comparations seems to work
  // minimum positive subnormal number 00000001 1.40129846e-45
  _tmp1 = (_MM_TYPE)_MM_CMPEQ_SIG((_MM_TYPE_I)_MM_LOADU((fptype*)otype), (_MM_TYPE_I)_MM_SETZERO()); // otype ? // FIXME FOR DOUBLE fptype
  _answer = _MM_OR(_MM_AND(_tmp1, _c), _MM_ANDNOT(_tmp1, _p));

  _MM_STORE(OptionPrice, _answer);

}

#endif // SIMD_WIDTH

/* JMCG END */

fptype CNDF ( fptype InputX )
{
  int sign;

  fptype OutputX;
  fptype xInput;
  fptype xNPrimeofX;
  fptype expValues;
  fptype xK2;
  fptype xK2_2, xK2_3;
  fptype xK2_4, xK2_5;
  fptype xLocal, xLocal_1;
  fptype xLocal_2, xLocal_3;

  // Check for negative value of InputX
  if (InputX < 0.0) {
    InputX = -InputX;
    sign = 1;
  } else
    sign = 0;

  xInput = InputX;

  // Compute NPrimeX term common to both four & six decimal accuracy calcs
  expValues = exp(-0.5f * InputX * InputX);
  xNPrimeofX = expValues;
  xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

  xK2 = 0.2316419 * xInput;
  xK2 = 1.0 + xK2;
  xK2 = 1.0 / xK2;
  xK2_2 = xK2 * xK2;
  xK2_3 = xK2_2 * xK2;
  xK2_4 = xK2_3 * xK2;
  xK2_5 = xK2_4 * xK2;

  xLocal_1 = xK2 * 0.319381530;
  xLocal_2 = xK2_2 * (-0.356563782);
  xLocal_3 = xK2_3 * 1.781477937;
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_4 * (-1.821255978);
  xLocal_2 = xLocal_2 + xLocal_3;
  xLocal_3 = xK2_5 * 1.330274429;
  xLocal_2 = xLocal_2 + xLocal_3;

  xLocal_1 = xLocal_2 + xLocal_1;
  xLocal   = xLocal_1 * xNPrimeofX;
  xLocal   = 1.0 - xLocal;

  OutputX  = xLocal;

  if (sign) {
    OutputX = 1.0 - OutputX;
  }

  return OutputX;
}

fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype, float timet )
{
  fptype OptionPrice;

  // local private working variables for the calculation
  fptype xStockPrice;
  fptype xStrikePrice;
  fptype xRiskFreeRate;
  fptype xVolatility;
  fptype xTime;
  fptype xSqrtTime;

  fptype logValues;
  fptype xLogTerm;
  fptype xD1;
  fptype xD2;
  fptype xPowerTerm;
  fptype xDen;
  fptype d1;
  fptype d2;
  fptype FutureValueX;
  fptype NofXd1;
  fptype NofXd2;
  fptype NegNofXd1;
  fptype NegNofXd2;

  xStockPrice = sptprice;
  xStrikePrice = strike;
  xRiskFreeRate = rate;
  xVolatility = volatility;

  xTime = time;
  xSqrtTime = sqrt(xTime);

  logValues = log( sptprice / strike );

  xLogTerm = logValues;


  xPowerTerm = xVolatility * xVolatility;
  xPowerTerm = xPowerTerm * 0.5;

  xD1 = xRiskFreeRate + xPowerTerm;
  xD1 = xD1 * xTime;
  xD1 = xD1 + xLogTerm;

  xDen = xVolatility * xSqrtTime;
  xD1 = xD1 / xDen;
  xD2 = xD1 -  xDen;

  d1 = xD1;
  d2 = xD2;

  NofXd1 = CNDF( d1 );
  NofXd2 = CNDF( d2 );

  FutureValueX = strike * ( exp( -(rate)*(time) ) );
  if (otype == 0) {
    OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
  } else {
    NegNofXd1 = (1.0 - NofXd1);
    NegNofXd2 = (1.0 - NofXd2);
    OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
  }

  return OptionPrice;
}


#ifdef ENABLE_TBB
struct mainWork {
  mainWork(){}
  mainWork(mainWork &w, tbb::split){}
/* JMCG */
#ifdef SIMD_WIDTH
  void operator()(const tbb::blocked_range<int> &range) const {
    _MM_ALIGN fptype price[SIMD_WIDTH];
    fptype priceDelta;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i+=SIMD_WIDTH) {
      /* Calling main function to calculate option value based on
       * Black & Scholes's equation.
       */

      BlkSchlsEqEuroNoDiv_SIMD( price, SIMD_WIDTH, &(sptprice[i]), &(strike[i]),
                           &(rate[i]), &(volatility[i]), &(otime[i]),
                           &(otype[i]), 0);
      for (int k=0; k<SIMD_WIDTH; k++) {
        prices[i+k] = price[k];

#ifdef ERR_CHK
        priceDelta = data[i+k].DGrefval - price[k];
        if( fabs(priceDelta) >= 1e-5 ){
          fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                 i+k, price, data[i+k].DGrefval, priceDelta);
          numError ++;
        }
#endif
      }
    }
  }

#else // !SIMD_WIDTH

  void operator()(const tbb::blocked_range<int> &range) const {
    fptype price;
    int begin = range.begin();
    int end = range.end();

    for (int i=begin; i!=end; i++) {
      /* Calling main function to calculate option value based on
       * Black & Scholes's equation.
       */

      price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                   rate[i], volatility[i], otime[i],
                                   otype[i], 0);
      prices[i] = price;

#ifdef ERR_CHK
      fptype priceDelta = data[i].DGrefval - price;
      if( fabs(priceDelta) >= 1e-5 ){
        fprintf(stderr,"Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
		i, price, data[i].DGrefval, priceDelta);
        numError ++;
      }
#endif
    }
  }
#endif // SIMD_WIDTH
/* JMCG END */
};

#endif // ENABLE_TBB

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_TBB
int bs_thread(void *tid_ptr) {
    int j;
    tbb::affinity_partitioner a;

    mainWork doall;
    for (j=0; j<NUM_RUNS; j++) {
      tbb::parallel_for(tbb::blocked_range<int>(0, numOptions), doall, a);
    }

    return 0;
}
#else // !ENABLE_TBB

#ifdef WIN32
DWORD WINAPI bs_thread(LPVOID tid_ptr){
#else
int bs_thread(void *tid_ptr) {
#endif
/* JMCG */
#ifdef SIMD_WIDTH
    int i, j, k;
    _MM_ALIGN fptype price[SIMD_WIDTH];
    fptype priceDelta;
    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions / nThreads);
    int end = start + (numOptions / nThreads);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_thread_begin();
#endif

    for (j=0; j<NUM_RUNS; j++) {
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i += SIMD_WIDTH) {
#else  //ENABLE_OPENMP
        for (i=start; i<end; i += SIMD_WIDTH) {
#endif //ENABLE_OPENMP
            // Calling main function to calculate option value based on Black & Scholes's
            // equation.
            BlkSchlsEqEuroNoDiv_SIMD(price, SIMD_WIDTH, &(sptprice[i]), &(strike[i]),
                                &(rate[i]), &(volatility[i]), &(otime[i]), &(otype[i]), 0);
            for (k=0; k<SIMD_WIDTH; k++) {
              prices[i+k] = price[k];
            }
#ifdef ERR_CHK
            for (k=0; k<SIMD_WIDTH; k++) {
                priceDelta = data[i+k].DGrefval - price[k];
                if (fabs(priceDelta) >= 1e-4) {
                    printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
                           i + k, price[k], data[i+k].DGrefval, priceDelta);
                    numError ++;
                }
            }
#endif
        }
    }

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_thread_end();
#endif

    return 0;
}
#else // ! SIMD_WIDTH
      int i, j;
      fptype price;
      fptype priceDelta;
      int tid = *(int *)tid_ptr;
      int start = tid * (numOptions / nThreads);
      int end = start + (numOptions / nThreads);

#ifdef ENABLE_PARSEC_HOOKS
      __parsec_thread_begin();
#endif

      for (j=0; j<NUM_RUNS; j++) {
#ifdef ENABLE_OPENMP
#pragma omp parallel for private(i, price, priceDelta)
        for (i=0; i<numOptions; i++) {
#else  //ENABLE_OPENMP
	  for (i=start; i<end; i++) {
#endif //ENABLE_OPENMP
            /* Calling main function to calculate option value based on
             * Black & Scholes's equation.
             */
            price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                         rate[i], volatility[i], otime[i],
                                         otype[i], 0);
            prices[i] = price;

#ifdef ERR_CHK
            priceDelta = data[i].DGrefval - price;
            if( fabs(priceDelta) >= 1e-4 ){
	      printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
		     i, price, data[i].DGrefval, priceDelta);
	      numError ++;
            }
#endif
	  }
	}

#ifdef ENABLE_PARSEC_HOOKS
        __parsec_thread_end();
#endif

	return 0;
      }
#endif // SIMD_WIDTH
/* JMCG END */
#endif //ENABLE_TBB

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    fptype * buffer;
    int * buffer2;
    int rv;

#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
        printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
	fflush(NULL);
#else
        printf("PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_blackscholes);
#endif

   if (argc != 4)
        {
                printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
                exit(1);
        }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    //Read input data from file
    file = fopen(inputFile, "r");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", inputFile);
      exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if(rv != 1) {
      printf("ERROR: Unable to read from file `%s'.\n", inputFile);
      fclose(file);
      exit(1);
    }

    /* JMCG */
#ifdef SIMD_WIDTH
    if(SIMD_WIDTH > numOptions) {
      printf("ERROR: Not enough work for SIMD operation.\n");
      fclose(file);
      exit(1);
    }
    if(nThreads > numOptions/SIMD_WIDTH) {
      printf("WARNING: Not enough work, reducing number of threads to match number of SIMD options packets.\n");
      nThreads = numOptions/SIMD_WIDTH;
    }
#endif
    /* JMCG END */
#if !defined(ENABLE_THREADS) && !defined(ENABLE_OPENMP) && !defined(ENABLE_TBB)
    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version)\n");
        exit(1);
    }
#endif

    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
    {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if(rv != 9) {
          printf("ERROR: Unable to read from file `%s'.\n", inputFile);
          fclose(file);
          exit(1);
        }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", inputFile);
      exit(1);
    }

#ifdef ENABLE_THREADS
    MAIN_INITENV(,8000000,nThreads);
#endif
    printf("Num of Options: %d\n", numOptions);
    printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;
        otime[i]      = data[i].t;
    }

    printf("Size of data: %d\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

#ifdef ENABLE_THREADS
#ifdef WIN32
    HANDLE *threads;
    int *nums;
    threads = (HANDLE *) malloc (nThreads * sizeof(HANDLE));
    nums = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        nums[i] = i;
        threads[i] = CreateThread(0, 0, bs_thread, &nums[i], 0, 0);
    }
    WaitForMultipleObjects(nThreads, threads, TRUE, INFINITE);
    free(threads);
    free(nums);
#else
    int *tids;
    tids = (int *) malloc (nThreads * sizeof(int));

    for(i=0; i<nThreads; i++) {
        tids[i]=i;
        CREATE_WITH_ARG(bs_thread, &tids[i]);
    }
    WAIT_FOR_END(nThreads);
    free(tids);
#endif //WIN32
#else //ENABLE_THREADS
#ifdef ENABLE_OPENMP
    {
        int tid=0;
        omp_set_num_threads(nThreads);
        bs_thread(&tid);
    }
#else //ENABLE_OPENMP
#ifdef ENABLE_TBB
    tbb::task_scheduler_init init(nThreads);

    int tid=0;
    bs_thread(&tid);
#else //ENABLE_TBB
    //serial version
    int tid=0;
    bs_thread(&tid);
#endif //ENABLE_TBB
#endif //ENABLE_OPENMP
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

    //Write prices to output file
    file = fopen(outputFile, "w");
    if(file == NULL) {
      printf("ERROR: Unable to open file `%s'.\n", outputFile);
      exit(1);
    }
    rv = fprintf(file, "%i\n", numOptions);
    if(rv < 0) {
      printf("ERROR: Unable to write to file `%s'.\n", outputFile);
      fclose(file);
      exit(1);
    }
    for(i=0; i<numOptions; i++) {
      rv = fprintf(file, "%.18f\n", prices[i]);
      if(rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
      }
    }
    rv = fclose(file);
    if(rv != 0) {
      printf("ERROR: Unable to close file `%s'.\n", outputFile);
      exit(1);
    }

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif
    free(data);
    free(prices);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}

