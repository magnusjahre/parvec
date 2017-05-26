#include "globals.h"
#include "functions.h"
#include "pthread_funcs.h"

//Initializes a struct array so its thread can have its own necessary data
thread_data_t* setup_thread_data(int num_threads, int imax, int jmax){

	thread_data_t * tdata;
	tdata = (thread_data_t*)malloc(num_threads * sizeof(thread_data_t));

	for(int i=0;i<num_threads;i++)
	{
		tdata[i].thread_id = i;
		tdata[i].numThreads = num_threads;
		tdata[i].imax=imax;
		tdata[i].jmax=jmax;	
	}

	return tdata;
}

//the actual algorithm which will be executed by each thread
void *p_SmithWaterman(void * ptr_to_tdata){
	thread_data_t * td = (thread_data_t*)ptr_to_tdata;
	float temp[4];
	int i,j,wave,tStart,tEnd;
	array_max_t arraymax;

	tStart= ((td->jmax/td->numThreads)*td->thread_id)+1;
	tEnd= tStart+(td->jmax/td->numThreads) ;
	//printf("Hello from thread: %d Numthreads:%d -%d -%d -%d -!\n",tID,numThreads,tStart,tEnd,N_b);
	for(wave=1; wave<=td->imax+td->numThreads-1; wave++) {
		i=wave-td->thread_id;
		if(i>=1 && i<=td->imax)
		{
			for(j=tStart; j<=tEnd; j++) {
			    temp[0] = H[i-1][j-1]+similarity_score(seq_a[i-1],seq_b[j-1]);
			    temp[1] = H[i-1][j]-delta;
			    temp[2] = H[i][j-1]-delta;
			    temp[3] = 0.;
				arraymax= find_array_max(temp, 4);
			    H[i][j] = arraymax.max;
			    switch(arraymax.ind) {
			    case 0:                                  // score in (i,j) stems from a match/mismatch
			        I_i[i][j] = i-1;
			        I_j[i][j] = j-1;
			        break;
			    case 1:                                  // score in (i,j) stems from a deletion in sequence A
			        I_i[i][j] = i-1;
			        I_j[i][j] = j;
			        break;
			    case 2:                                  // score in (i,j) stems from a deletion in sequence B
			        I_i[i][j] = i;
			        I_j[i][j] = j-1;
			        break;
			    case 3:                                  // (i,j) is the beginning of a subsequence
			        I_i[i][j] = i;
			        I_j[i][j] = j;
			        break;
			    }
			}
		}
		pthread_barrier_wait(&bar);			
	}
	return ((void*) 0);
}

//parallized function searches for the highest score in the matrix
void *search_max_score(void* ptr_data){
	thread_data_t * td = (thread_data_t*)ptr_data;
	float tH_max = 0.;
    int ti_max=0,tj_max=0,tStart,tEnd;

	tStart= ((td->imax/td->numThreads)*td->thread_id)+1;
	tEnd= tStart+(td->imax/td->numThreads) ;
	if(tEnd>td->imax)tEnd=td->imax;
	//each thread searches a certain number of lines
    for(int i=tStart; i<=tEnd; i++) {
        for(int j=1; j<=td->jmax; j++) {
            if(H[i][j]>tH_max) {
                tH_max = H[i][j];
                ti_max = i;
                tj_max = j;
            }
        }
    }
	//compare with the global highest scores
	pthread_mutex_lock(&mutex);
    if(tH_max>H_max) {
        H_max = tH_max;
        i_max = ti_max;
        j_max = tj_max;
    }
	pthread_mutex_unlock(&mutex);
	return ((void*) 0);
}
