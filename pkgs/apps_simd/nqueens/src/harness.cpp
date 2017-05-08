#include "harness.h"

#include <pthread.h>
//#include "papiprofiler.h"

const float min_stability = 2.0f;

int Harness::numt;
bool Harness::sort_flag;
bool Harness::verbose;
int Harness::block_size;
int Harness::splice_depth;
struct timeval Harness::start_time[num_counters], Harness::end_time[num_counters];
uint64_t Harness::sum_time[num_counters];
string Harness::benchmark;
string Harness::appargs;
int Harness::autotuneBlock, Harness::autotuneDepth, Harness::autotuneFraction;

// harness params for ParallelismProfiler
int Harness::vectorWidth, Harness::truncateCost, Harness::recurseCost;
float Harness::autotuneReach;

#ifdef PAPI
PapiProfiler Harness::papi_profiler[num_counters];
#endif

int Harness::get_elapsed_usec(int i) {
	int sec = end_time[i].tv_sec - start_time[i].tv_sec;
	int usec = end_time[i].tv_usec - start_time[i].tv_usec;
	return (sec * 1000000) + usec;
}

int *distribute_among(int initStart, int end, int numt) {
	int *ret = new int[numt * 2];
	int start = initStart;
	int i;
	int num;
	for (i = 0; i < numt; i++) {
		num = (end - start) / (numt - i);
		ret[2 * i] = start;
		ret[2 * i + 1] = start + num;
		start += num;
	}
	return ret;
}

struct targs {
	thread_function func;
	int start, end;
};

static void *thread_entry(void * arg) {
	struct targs * t = (struct targs*)arg;
	t->func(t->start, t->end);
  return NULL;
}

void Harness::start_timing(int i) {
	gettimeofday(&start_time[i], NULL);
#ifdef PAPI
	if (i == 0) papi_profiler[i].start();
#endif
}

void Harness::stop_timing(int i) {
	gettimeofday(&end_time[i], NULL);
	sum_time[i] += get_elapsed_usec(i);
#ifdef PAPI
	if (i == 0) papi_profiler[i].stop();
#endif
}

/* schedules a thread function to run for a particular number of threads */
void Harness::parallel_for(thread_function func, int start, int end) {

	uint32_t i;
	struct targs *args = NULL;
	pthread_t *threads = NULL;
	int *ranges;

	if (numt == 1) {
		func(start, end);
	} else {
		threads = new pthread_t[numt];
		if(!threads) {
			fprintf(stderr, "error: could not allocate threads.\n");
			exit(1);
		}

		args = new struct targs[numt];
		if(!args) {
			fprintf(stderr, "error: could not allocate thread args.\n");
			exit(1);
		}

		ranges = distribute_among(start, end, numt);
		for(i = 0; i < numt; i++) {
			args[i].start = ranges[2 * i];
			args[i].end = ranges[2 * i + 1];
			args[i].func = func;
			if(pthread_create(&threads[i], NULL, thread_entry, &args[i]) != 0) {
				fprintf(stderr, "error: could not create thread.\n");
			}
		}
		delete [] ranges;

		for(i = 0; i < numt; i++) {
			pthread_join(threads[i], NULL);
		}

		delete [] threads;
		delete [] args;
	}
}

float get_mean_min_max(int start, int end, int *runtimes, int *min, int *max) {
	int runtimes_sum = 0;
	int runtimes_max = 0;
	int runtimes_min = 1 << 30;
	int i;
	float runtimes_avg;

	for (i = start; i < end; i++) {
		runtimes_sum += runtimes[i];
		runtimes_min = runtimes_min < runtimes[i] ? runtimes_min : runtimes[i];
		runtimes_max = runtimes_max > runtimes[i] ? runtimes_max : runtimes[i];
	}
	runtimes_avg = (float) runtimes_sum / (end - start);
	if (min != NULL) *min = runtimes_min;
	if (max != NULL) *max = runtimes_max;
	return runtimes_avg;
}

float get_mean(int start, int end, int *runtimes) {
	int runtimes_sum = 0;
	float runtimes_avg;
	for (int i = start; i < end; i++) {
		runtimes_sum += runtimes[i];
	}
	runtimes_avg = (float) runtimes_sum / (end - start);
	return runtimes_avg;
}

float get_mean_float(int start, int end, float *runtimes) {
	float runtimes_sum = 0;
	float runtimes_avg;
	for (int i = start; i < end; i++) {
		runtimes_sum += runtimes[i];
	}
	runtimes_avg = runtimes_sum / (end - start);
	return runtimes_avg;
}

float get_stability_stdev(int start, int end, int *runtimes, float mean, float *stdev) {
	float runtimes_variance_sum = 0;
	for (int i = start; i < end; i++) {
		runtimes_variance_sum = (runtimes[i] - mean) * (runtimes[i] - mean);
	}
	if (end - start > 1) runtimes_variance_sum = runtimes_variance_sum / (end - start - 1);
	else runtimes_variance_sum = 0;
	float runtimes_stdev = sqrt(runtimes_variance_sum);
	float runtimes_stability = runtimes_stdev * 100 / mean;
	if (stdev) *stdev = runtimes_stdev;
	return runtimes_stability;
}

extern int app_main(int argc, char **argv);

int main(int argc, char **argv) {
	return Harness::run(argc, argv);
}

void Harness::init() {
	numt = 1;
	sort_flag = 0;
	verbose = 0;
	block_size = 128;
	splice_depth = 4;
	autotuneFraction = 100;

	// harness params for ParallelismProfiler
	vectorWidth = 4;
	truncateCost = 1;
	recurseCost = 1;
}

int Harness::run(int argc, char **argv) {

	init();

	int runs = 1;
	int drop_runs = 0;
	int max_runs = 0;
	int appargc = 0;
	char **appargv;
	char *arg;


	int papi_arg = 0;
	for (int i = 1; i < argc; i++) {
		arg = argv[i];
		if (strcmp(arg, "-t") == 0) {
			numt = atoi(argv[++i]);
			if(numt <= 0) {
				fprintf(stderr, "error: invalid number of threads.\n");
				exit(1);
			}
		} else if (strcmp(arg, "-r") == 0) {
			runs = atoi(argv[++i]);
			if(runs <= 0) {
				fprintf(stderr, "error: invalid number of runs.\n");
				exit(1);
			}
		} else if (strcmp(arg, "--dr") == 0) {
			drop_runs = atoi(argv[++i]);
			if(drop_runs < 0) {
				fprintf(stderr, "error: invalid number of drops.\n");
				exit(1);
			}
		} else if (strcmp(arg, "--mr") == 0) {
			max_runs = atoi(argv[++i]);
			if(max_runs < 0 || max_runs < runs) {
				fprintf(stderr, "error: invalid number of max runs.\n");
				exit(1);
			}
		} else if (strcmp(arg, "-s") == 0) {
			sort_flag = 1;
		} else if (strcmp(arg, "-v") == 0) {
			verbose = 1;
		} else if (strcmp(arg, "--autotuneFraction") == 0) {
			autotuneFraction = atoi(argv[++i]);
#ifdef PARALLELISM_PROFILE
		} else if (strcmp(arg, "--vectorWidth") == 0) {
			vectorWidth = atoi(argv[++i]);
		} else if (strcmp(arg, "--truncateCost") == 0) {
			truncateCost = atoi(argv[++i]);
		} else if (strcmp(arg, "--recurseCost") == 0) {
			recurseCost = atoi(argv[++i]);
#endif
		} else if (strcmp(arg, "--block") == 0) {
			block_size = atoi(argv[++i]);
		} else if (strcmp(arg, "--splice") == 0) {
			splice_depth = atoi(argv[++i]);
#ifdef PAPI
		} else if (strcmp(arg, "--papi") == 0) {
			PapiProfiler::set_arg(atoi(argv[++i]));
#endif
		} else if (strcmp(arg, "--appargs") == 0) {
			/*
			 * some apps like uts (unbalanced tree search) have overlapping arg specifiers with Harness
			 * e.g., "-t 1 -a 3 -d 10 -b 4 -r 19"
			 * allow explicit passing of args to app, with --appargs
			 * e.g., ./uts -t 2 --appargs -t 1 -a 3 -d 10 -b 4 -r 19
			 * first -t 2 specifies 2 threads, second -t 1 passed to app
			 */
			appargc = argc - i;
			appargv = argv + i;
			break;
		} else {
			appargc = argc - i;
			appargv = argv + i;
			break;
		}
	}
	if (max_runs == 0) max_runs = (runs - drop_runs) * 3 + drop_runs;

	benchmark = string(argv[0]);
#ifdef ICC
	benchmark += "_icc";
#endif
#ifdef AUTOVEC
	benchmark += "_autovec";
#endif
#ifdef AVX
	benchmark += "_avx";
#endif
#ifdef VEC_COPY
	benchmark += "_veccopy";
#endif
#ifdef SIMD_ALL
	benchmark += "_all";
#endif
#ifdef SIMD_NONE
	benchmark += "_none";
#endif
#ifdef SIMD_OPPORTUNITY
	benchmark += "_opportunity";
#endif
#ifdef NO_ELISION
	benchmark += "_noelision";
#endif
#ifdef BLOCK_TOP
	benchmark += "_blocktop";
#endif
#ifdef ONE_PHASE
	benchmark += "_onephase";
#endif

	cout << benchmark << endl;

	printf("args:");
	for (int i = 0; i < argc; i++) {
		printf("%s ", argv[i]);
	}
	printf("\n");
	printf("[numt:%d][runs:%d][dropruns:%d][maxruns:%d][sort_flag:%d][verbose:%d][block:%d][splice:%d][vectorWidth:%d]\n",
			numt, runs, drop_runs, max_runs, sort_flag, verbose, block_size, splice_depth, vectorWidth);
	printf("appargc: %d\n", appargc);

	appargs = "";
	for (int i = 0; i < appargc; i++) {
		appargs += (string(appargv[i]) + " ");
	}
	cout << "appargs: " << appargs << endl;


	int **runtimes = new int*[num_counters];
	int *autotuneBlocks = new int[max_runs];
	int *autotuneDepths = new int[max_runs];
	float *autotuneReachs = new float[max_runs];
	for (int i = 0; i < num_counters; i++) {
		runtimes[i] = new int[max_runs];
		for (int j = 0; j < max_runs; j++) {
			runtimes[i][j] = 0;
		}
	}
	if(!runtimes) {
		fprintf(stderr, "invalid num recorded_runs\n");
		exit(1);
	}

	// invoke benchmark multiple times
	int run = 0;
	for (; run < max_runs; run++) {
		if (run >= runs) {
			if (runs == 1) break;
			bool is_stable = true;
			float stability;
			for (int i = 0; i < num_counters; i++) {
				float mean = get_mean_min_max(drop_runs, run, runtimes[i], NULL, NULL);
				stability = get_stability_stdev(drop_runs, run, runtimes[i], mean, NULL);
				if (stability > min_stability) {
					is_stable = false;
					break;
				}
			}
			if (is_stable) break;
			else {
				drop_runs++;
				printf("unstable [run:%d][dropruns:%d][stability:%.2f]\n", run, drop_runs, stability);
			}
		}

		for (int i = 0; i < num_counters; i++) {
			sum_time[i] = 0;
		}
		autotuneBlock = 0;
		autotuneDepth = 0;

		app_main(appargc, appargv);

		printf("[run:%d]", run);
		for (int i = 0; i < num_counters; i++) {
			runtimes[i][run] = sum_time[i] / 1000;
			printf("[runtime%d:%d]", i, runtimes[i][run]);
		}
		autotuneBlocks[run] = autotuneBlock;
		autotuneDepths[run] = autotuneDepth;
		autotuneReachs[run] = autotuneReach;
		printf("[autotune block:%d depth:%d]", autotuneBlock, autotuneDepth);
		printf("\n");
	}
	int actual_runs = run;
	if (actual_runs == max_runs) {drop_runs++;}
	//printf("start writing files...\n");

	bool print_header = false;
	FILE *fp = fopen("stats.csv", "r");
	if (fp == NULL) {
		print_header = true;
	} else {
		fclose(fp);
	}

	fp = fopen("stats.csv", "a");
	assert(fp != NULL);
	if (print_header) {
		fprintf(fp, "benchmark, args, sort_flag, actualruns, dropruns, numt, blocksize, avg, min, max, stdev, stability\n");
	}


	fprintf(fp, "%s,%s,   %d,%d,%d,%d,%d,%d, ",
			benchmark.c_str(), appargs.c_str(),
			sort_flag, actual_runs, drop_runs, numt, block_size, splice_depth);
	float avgAutotuneBlock = get_mean(drop_runs, actual_runs, autotuneBlocks);
	float avgAutotuneDepth = get_mean(drop_runs, actual_runs, autotuneDepths);
	float avgAutotuneReach = get_mean_float(drop_runs, actual_runs, autotuneReachs);
	fprintf(fp, "%.2f,%.2f,%.2f, ", avgAutotuneBlock, avgAutotuneDepth, avgAutotuneReach);
	float runtimes_avg0;
	for (int i = 0; i < num_counters; i++) {
		int runtimes_min, runtimes_max;
		float runtimes_stdev;
		float runtimes_avg = get_mean_min_max(drop_runs, actual_runs, runtimes[i], &runtimes_min, &runtimes_max);
		if (i == 0) runtimes_avg0 = runtimes_avg;
		float runtimes_stability = get_stability_stdev(drop_runs, actual_runs, runtimes[i], runtimes_avg, &runtimes_stdev);
		if (runtimes_max == 0) break;
		fprintf(fp, "%.2f, %d, %d, %.2f, %.2f, ", runtimes_avg, runtimes_min, runtimes_max, runtimes_stdev, runtimes_stability);
		printf("[%d][runtime:%.2f(msec)][min:%d][max:%d][stability:%.2f]\n",
				i, runtimes_avg, runtimes_min, runtimes_max, runtimes_stability);
	}
	fprintf(fp, "\n");
	fclose(fp);

	fp = fopen("fullstats.csv", "a");
	assert(fp != NULL);

	for (int i = 0; i < actual_runs; i++) {
		fprintf(fp, "%s,%s,   %d,%d,%d,%d,%d,%d, %d, ",
				benchmark.c_str(), appargs.c_str(),
				sort_flag, runs, drop_runs, numt, block_size, splice_depth,
				i);
		fprintf(fp, "%d,%d,%d, ", autotuneBlocks[i], autotuneDepths[i], autotuneReachs[i]);
		for (int j = 0; j < num_counters; j++) {
			fprintf(fp, "%d, ", runtimes[j][i]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

#ifdef PAPI
	papi_profiler[0].output(drop_runs, actual_runs, runtimes_avg0, avgAutotuneBlock, avgAutotuneDepth);
#endif

	for (int i = 0; i < num_counters; i++) {
		delete [] runtimes[i];
	}
	delete [] runtimes;
	delete [] autotuneBlocks;
	delete [] autotuneDepths;

	return 0;
}



