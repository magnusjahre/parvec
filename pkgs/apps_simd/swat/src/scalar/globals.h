#ifndef GLOBALS_H
#define GLOBALS_H

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <sys/time.h>
#include <pthread.h>

using namespace std;

extern unsigned short **I_i, **I_j;
extern float mu, delta, **H;
extern string seq_a,seq_b;

extern float H_max;
extern int i_max, j_max;

extern struct timeval	StartTime, EndTime;

struct array_max_t{
	int ind;
	float max;
};

struct thread_data_t{
	int thread_id, numThreads, imax, jmax;
};

/*defines the pthread barrier*/
extern pthread_barrier_t bar;

/*defines the pthread mutex*/
extern pthread_mutex_t mutex;

#endif
