#include "globals.h"

unsigned short **I_i, **I_j;
float mu, delta, **H;
string seq_a,seq_b;

float H_max=0.;
int i_max=0, j_max=0;

struct timeval	StartTime, EndTime;

/*defines the pthread barrier*/
pthread_barrier_t bar;

/*defines the pthread mutex*/
pthread_mutex_t mutex;
