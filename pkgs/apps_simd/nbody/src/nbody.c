#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

/* CDF BEGIN */
#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Double precision
#ifdef DOUBLEPREC
#define DFTYPE	// NOTE: This ALWAYS needs to go before simd_defines.h!
typedef double real_t;
#endif

// Single precision
#ifdef SINGLEPREC
typedef float real_t;
#endif


#include "simd_defines.h"
/* CDF END */


// Algorithm parameters
#define G 0.1
#define STEPSIZE 0.0001
#define FORCELIMIT 0.0001


/************************/
/*				   		*/
/*	RUNTIME PARAMETERS	*/
/*				   		*/
/************************/
#define NBODIES atoi(argv[1])
#define NTIMESTEPS 2000

/************************/

// Build scalar when using gcc-hooks
#if !defined (PARSEC_USE_SSE) && !defined (PARSEC_USE_AVX)
#define SCALAR 1
#endif


// Function prototypes
void RunSimulation(const int ntimeSteps, const int nBodies);

void TimeStep(
	FILE * plotfile,
	const int timeStep,
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void ComputeAccel(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);
	
void ComputeAccelParvec(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass);

void UpdatePositions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay);

void SetInitialConditions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass);

void PrintPositions(FILE * plotfile, const int nBodies, const real_t * restrict rx, const real_t * restrict ry);
double ErrorCheck(const int nBodies, const real_t * restrict rx);

// Utility functions
double GetWallTime(void);




int main(int argc, char **argv)
{
	
	//const int n = &argv[1];

	
	
	// When the benchmark begins (main), use the same string "__parsec_streamcluster", it's just ignored
#ifdef ENABLE_PARSEC_HOOKS
	 __parsec_bench_begin(__parsec_streamcluster);
#endif

/*
	for (int nBodies = 10; nBodies < 200; nBodies += 10) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 100; nBodies < 300; nBodies += 20) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 300; nBodies < 1000; nBodies += 100) {
		RunSimulation(nTimeSteps, nBodies);
	}
	for (int nBodies = 1000; nBodies <= 1000; nBodies += 500) {
		RunSimulation(nTimeSteps, nBodies);
	}
*/
	RunSimulation(NTIMESTEPS, NBODIES);
	printf("Complete!\n[EXECUTABLE] Method: %d bodies, ", NBODIES);
#if defined (SCALAR)
	printf("SCALAR config ");
#elif defined (PARSEC_USE_AVX)
	printf("PARVEC WRAPPER with AVX config");
#elif defined (PARSEC_USE_SSE)
	printf("PARVEC WRAPPER with SSE config");
#else
	printf("UNDEFINED\n");
#endif
#if defined (SINGLEPREC)
	printf("using SINGLE PRECISION\n");
#elif defined (DOUBLEPREC)
	printf("using DOUBLE PRECISION\n");
#endif


	// Before you exit the application
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end(); 
#endif
	
	return 0;
}


void RunSimulation(const int nTimeSteps, const int nBodies)
{
	double timeElapsed;

	// files for printing
	char filename[25];
	sprintf(filename, "pos%d.dat", nBodies);
	FILE * datfile = fopen(filename,"w");
	FILE * plotfile = fopen("plot.plt","w");

	// Allocate arrays
	real_t * rx;
	real_t * ry;
	real_t * vx;
	real_t * vy;
	real_t * ax;
	real_t * ay;
	real_t * mass;
	rx =   _mm_malloc(nBodies * sizeof *rx,32);
	ry =   _mm_malloc(nBodies * sizeof *ry,32);
	vx =   _mm_malloc(nBodies * sizeof *vx,32);
	vy =   _mm_malloc(nBodies * sizeof *vy,32);
	ax =   _mm_malloc(nBodies * sizeof *ax,32);
	ay =   _mm_malloc(nBodies * sizeof *ay,32);
	mass = _mm_malloc(nBodies * sizeof *mass,32);

	SetInitialConditions(nBodies, rx,ry, vx,vy, ax,ay, mass);



#ifdef PRINTPOS
	fprintf(plotfile, "set term pngcairo enhanced size 1024,768\n");
	fprintf(plotfile, "set output \"test.png\"\n");
	fprintf(plotfile, "set grid\n");
	fprintf(plotfile, "set key off\n");
	fprintf(plotfile, "set xrange [-100:100]\n");
	fprintf(plotfile, "set yrange [-100:100]\n");
	fprintf(plotfile, "plot \\\n");
	for (int i = 0; i < nBodies-1; i++) {
		fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints,\\\n", 2*i+1, 2*i+2);
	}
	fprintf(plotfile, "\"pos.dat\" using %d:%d with linespoints\n", 2*(nBodies-1)+1, 2*(nBodies-1)+2);
#endif

	timeElapsed = GetWallTime();
	
	//  Right before the section of the code we want to measure (as we discussed, in your code, right AFTER "timeElapsed = GetWallTime();")
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif
	
	for (int n = 0; n < nTimeSteps; n++) {
		TimeStep(datfile, n, nBodies, rx,ry, vx,vy, ax,ay, mass);
	}

	// When we are done measuring, (for nbody, right BEFORE timeElapsed = GetWallTime() - timeElapsed;)
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

	timeElapsed = GetWallTime() - timeElapsed;
//	printf("nBodies: %4d, MegaUpdates/second: %lf. Error: %le\n", nBodies, nTimeSteps*nBodies/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));
	printf("%4d Time %lf MTSps %le Sumx %.15le\n", nBodies, timeElapsed, nTimeSteps/timeElapsed/1000000.0, ErrorCheck(nBodies, rx));



	_mm_free(rx);
	_mm_free(ry);
	_mm_free(vx);
	_mm_free(vy);
	_mm_free(ax);
	_mm_free(ay);
	_mm_free(mass);
	fclose(plotfile);
	fclose(datfile);


}


void TimeStep(
	FILE * plotfile,
	const int timeStep,
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{

//#ifdef ENABLE_PARSEC_HOOKS	
//	__parsec_thread_begin(); 
//#endif

#ifdef SCALAR
	//printf("SCALAR");
	ComputeAccel(nBodies, rx,ry, ax,ay, mass);
#else
	//printf("PARVEC WRAPPER");
	ComputeAccelParvec(nBodies, rx,ry, ax,ay, mass); 	// CDF PARVEC
#endif

	UpdatePositions(nBodies, rx,ry, vx,vy, ax,ay);

#ifdef PRINTPOS
	if (timeStep % 1000 == 0) {
		PrintPositions(plotfile, nBodies, rx,ry);
	}
#endif

//#ifdef ENABLE_PARSEC_HOOKS
//	__parsec_thread_end();
//#endif

}

#ifdef SCALAR
void ComputeAccel(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	
	double distx, disty, sqrtRecipDist;

	for (int i = 0; i < nBodies; i++) {
		for (int j = i+1; j < nBodies; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);

			// This version with a force-limiting term stops nearby bodies experiencing arbitrarily high
			// forces. Important for numerical stability, but not for performance testing.
//			ax[i] += (mass[j] * distx * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ay[i] += (mass[j] * disty * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ax[j] -= (mass[i] * distx * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
//			ay[j] -= (mass[i] * disty * pow(sqrtRecipDist*sqrtRecipDist+FORCELIMIT,3.0/2.0));
		}
	}

}
#endif



/* CDF PARVEC PORT BEGIN */

#ifndef SCALAR
void ComputeAccelParvec(
	const int nBodies,
	const real_t * restrict rx,
	const real_t * restrict ry,
	real_t * restrict ax,
	real_t * restrict ay,
	const real_t * restrict mass)
{
	double distx, disty, sqrtRecipDist;

	// For the wrapper library you have to use SIMD_WIDTH, 
	// that is set for the specific configuration you plan to run
	const int jVecMax = SIMD_WIDTH*(nBodies/SIMD_WIDTH);

	for (int i = 0; i < nBodies; i++) {

		// Vectorized j loop starts at multiple of 4 >= i+1
		const int jVecMin = (SIMD_WIDTH*((i)/SIMD_WIDTH)+SIMD_WIDTH) > nBodies ? nBodies : (SIMD_WIDTH*((i)/SIMD_WIDTH)+SIMD_WIDTH);

		// first initial non-vectorized part
		for (int j = i+1; j < jVecMin; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
		}

		// if we have already finished, break out of the loop early
		if (jVecMax < jVecMin) break;

		
		// Vectorized part
		_MM_TYPE rxiVec = _MM_SET(rx[i]);
		_MM_TYPE ryiVec = _MM_SET(ry[i]);
		_MM_TYPE massiVec = _MM_SET(mass[i]);
		_MM_TYPE axiUpdVec = _MM_SET(0.0);
		_MM_TYPE ayiUpdVec = _MM_SET(0.0);

		for (int j = jVecMin; j < jVecMax; j+=SIMD_WIDTH) {
			_MM_TYPE rxjVec = _MM_LOAD(&rx[j]);
			_MM_TYPE ryjVec = _MM_LOAD(&ry[j]);
			_MM_TYPE axjVec = _MM_LOAD(&ax[j]);
			_MM_TYPE ayjVec = _MM_LOAD(&ay[j]);
			_MM_TYPE massjVec = _MM_LOAD(&mass[j]);
			
			
			// *************** FAST MULTIPOLE METHOD ***************

			_MM_TYPE distxVec = _MM_SUB(rxiVec, rxjVec);
			_MM_TYPE distyVec = _MM_SUB(ryiVec, ryjVec);

#ifdef ORIGINAL
			// Original inv sqrt
			_MM_TYPE sqrtRecipDistVec = _MM_DIV(_MM_SET(1.0), _MM_SQRT(_MM_ADD(_MM_MUL(distxVec,distxVec), _MM_MUL(distyVec,distyVec))));
#endif
#ifdef IMPROVED			
			// Approximate inv sqrt (Single precision only)
			_MM_TYPE sqrtRecipDistVec = _MM_RSQRT( _MM_ADD(_MM_MUL(distxVec,distxVec), _MM_MUL(distyVec,distyVec)));
#endif

			// cube:
			sqrtRecipDistVec = _MM_MUL(sqrtRecipDistVec,
			_MM_MUL(sqrtRecipDistVec,sqrtRecipDistVec));

			// multiply into distxVec and distyVec
			distxVec = _MM_MUL(distxVec, sqrtRecipDistVec);
			distyVec = _MM_MUL(distyVec, sqrtRecipDistVec);

			// update accelerations
			axiUpdVec = _MM_ADD(axiUpdVec, _MM_MUL(massjVec, distxVec));
			ayiUpdVec = _MM_ADD(ayiUpdVec, _MM_MUL(massjVec, distyVec));
			axjVec = _MM_SUB(axjVec, _MM_MUL(massiVec, distxVec));
			ayjVec = _MM_SUB(ayjVec, _MM_MUL(massiVec, distyVec));
			
			
			// *****************************************************
			
			_MM_STORE(&ax[j], axjVec);
			_MM_STORE(&ay[j], ayjVec);
		}

		// Now need to sum elements of axiUpdVec,ayiUpdVec and add to ax[i],ay[i]
		axiUpdVec = _MM_HADD(axiUpdVec, axiUpdVec);
		ayiUpdVec = _MM_HADD(ayiUpdVec, ayiUpdVec);
		ax[i] += _MM_REDUCE_ADD(axiUpdVec);
		ay[i] += _MM_REDUCE_ADD(ayiUpdVec);



		// final non-vectorized part, iff we didn't already run up to a jVecMin which is larger than jVecMax
		for (int j = jVecMax; j < nBodies; j++) {
			distx = rx[i] - rx[j];
			disty = ry[i] - ry[j];
			sqrtRecipDist = 1.0/sqrt(distx*distx+disty*disty);
			ax[i] += (mass[j] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[i] += (mass[j] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ax[j] -= (mass[i] * distx * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
			ay[j] -= (mass[i] * disty * sqrtRecipDist*sqrtRecipDist*sqrtRecipDist);
		}
	}

}

/* CDF END */
#endif

void UpdatePositions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay)
{
	for (int i = 0; i < nBodies; i++) {
			//new force values in .ax, .ay
			//update pos and vel
			vx[i] += (-G)*STEPSIZE * ax[i];
			vy[i] += (-G)*STEPSIZE * ay[i];
			rx[i] += STEPSIZE * vx[i];
			ry[i] += STEPSIZE * vy[i];
			//zero accel values to avoid an extra loop in ComputeAccel
			ax[i] = 0;
			ay[i] = 0;
		}

}




void SetInitialConditions(
	const int nBodies,
	real_t * restrict rx,
	real_t * restrict ry,
	real_t * restrict vx,
	real_t * restrict vy,
	real_t * restrict ax,
	real_t * restrict ay,
	real_t * restrict mass)
{
/*
	// Set random initial conditions.
	for (int i = 0; i < nBodies; i++) {
		rx[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		ry[i] = ((double)rand()*(double)100/(double)RAND_MAX)*pow(-1,rand()%2);
		vx[i] = (rand()%3)*pow(-1,rand()%2);
		vy[i] = (rand()%3)*pow(-1,rand()%2);
		ax[i] = 0;
		ay[i] = 0;
		mass[i] = 1000;
	}
*/
	//Some deterministic initial conditions (for testing openmpi build's weird differences)
	for (int i = 0; i < nBodies; i++) {
		rx[i] = 500*i*pow(-1,i)*sin(i);
		ry[i] = -500*i*pow(-1,i)*cos(i);
		vx[i] = 10*i*i*pow(-1,i);
		vy[i] = -5*i*i*pow(-1,i);
		ax[i] = 0;
		ay[i] = 0;
		mass[i] = 1000;
	}
}

double ErrorCheck(const int nBodies, const real_t * restrict rx)
{
	// Compute sum of x coordinates. Can use to check consistency between versions.
	double sumx = 0;
	for (int i = 0; i < nBodies; i++) {
		sumx += rx[i];
	}
	return sumx;
}


void PrintPositions(FILE * file, const int nBodies, const real_t * restrict rx, const real_t * restrict ry)
{
	for(int i = 0; i < nBodies-1; i++) {
		fprintf(file, "%le %le ", rx[i],ry[i]);
	}
	fprintf(file, "%le %le\n",rx[nBodies-1],ry[nBodies-1]);
}





double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}
