/*
 * harness.h
 *
 *  Created on: May 30, 2012
 *      Author: yjo
 */

#ifndef HARNESS_H_
#define HARNESS_H_

#include "common.h"
#include <sys/time.h>

//#define TRACK_TRAVERSALS

typedef void (*thread_function)(int start, int end);

#ifdef PAPI
class PapiProfiler;
#endif

class Harness {
public:
	static bool get_sort_flag() { return sort_flag; }
	static bool get_verbose_flag() { return verbose; }
	static int get_block_size() { return block_size; }
	static void set_block_size(int b) { block_size = b; }
	static int get_splice_depth() { return splice_depth; }

	// harness params for ParallelismProfiler
	static int get_vectorWidth() { return vectorWidth; }
	static int get_truncateCost() { return truncateCost; }
	static int get_recurseCost() { return recurseCost; }

	static int get_autotuneFraction() { return autotuneFraction; }
	static string get_benchmark() { return benchmark; }
	static string get_appargs() { return appargs; }
	static void recordAutotuneParams(int b, int d, float r = 0) {
		autotuneBlock = b;
		autotuneDepth = d;
		autotuneReach = r;
	}

	static void start_timing() { start_timing(0); }
	static void stop_timing() { stop_timing(0); }
	static void start_timing(int i);
	static void stop_timing(int i);

	static int run(int argc, char **argv);
	static void parallel_for(thread_function func, int start, int end);

private:
	static const int num_counters = 4;

	static void init();

	static int get_elapsed_usec(int i);

	static int numt;
	static bool sort_flag;
	static bool verbose;
	static int block_size;
	static int splice_depth;
	static struct timeval start_time[num_counters], end_time[num_counters];
	static uint64_t sum_time[num_counters];
	static string benchmark;
	static string appargs;
	static int autotuneBlock, autotuneDepth;
	static float autotuneReach;
	static int autotuneFraction;

	// harness params for ParallelismProfiler
	static int vectorWidth, truncateCost, recurseCost;

#ifdef PAPI
public:
	//static PapiProfiler *get_papi_profiler(int i) { return &papi_profiler[i]; }
private:
	static PapiProfiler papi_profiler[num_counters];
#endif
};


#endif /* HARNESS_H_ */
