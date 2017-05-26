#ifndef PTHREAD_FUNCS_H
#define PTHREAD_FUNCS_H

thread_data_t* setup_thread_data(int num_threads, int imax, int jmax);

void *p_SmithWaterman(void * ptr_to_tdata);

void *search_max_score(void* ptr_data);

#endif
