#include "globals.h"
#include "functions.h"
#include "pthread_funcs.h"

/* CDF START */
//#ifdef ENABLE_PARSEC_HOOKS
#include "simd_defines.h"
#include <hooks.h>
//#endif
/* CDF END */

int main(int argc, char** argv) {
    
/* CDF START */
// When the benchmark begins (main), use the same string "__parsec_streamcluster", it's just ignored
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_begin(__parsec_streamcluster);
#endif
/* CDF END */

    // read info from arguments
    if(argc!=6) {
        cout<<"Give me the propen number of input arguments:"<<endl<<"1 : mu"<<endl;
        cout<<"2 : delta"<<endl<<"3 : filename sequence A"<<endl<<"4 : filename sequence B"<<endl;
        cout<<"5 : number of threads"<<endl;
        exit(1);
    }

    mu    = atof(argv[1]);
    delta = atof(argv[2]);

    char *nameof_seq_a = argv[3];
    char *nameof_seq_b = argv[4];
    int N_threads = atoi(argv[5]);
    //string seq_a,seq_b;

    // read the sequences into two vectors:
    ifstream stream_seq_a;
    stream_seq_a.open(nameof_seq_a);
    checkfile(! stream_seq_a,nameof_seq_a);
    cout << "Reading file \"" << nameof_seq_a << "\"\n";
    seq_a = read_sequence(stream_seq_a);
    cout << "File \"" << nameof_seq_a << "\" read\n\n";

    ifstream stream_seq_b;
    stream_seq_b.open(nameof_seq_b);
    checkfile(! stream_seq_b,nameof_seq_b);
    cout << "Reading file \"" << nameof_seq_b << "\"\n";
    seq_b = read_sequence(stream_seq_b);
    cout << "File \"" << nameof_seq_b << "\" read\n\n";

    int N_a = seq_a.length();
    int N_b = seq_b.length();

    cout << "First sequence has length  : " << setw(6) << N_a <<endl;
    cout << "Second sequence has length : " << setw(6) << N_b << endl << endl;


    cout << "Allocating memory for matrix H\n";
    H = (float **)malloc((N_a + 1) * sizeof(float *));
    if (H == NULL) {
        cout << "Could not allocate memory for matrix H\n";
        exit(1);
    }
    for (int i = 0; i < (N_a + 1); i++) {
        H[i] = (float *)malloc((N_b + 1) * sizeof(float));
        if (H[i] == NULL) {
            cout << "Could not allocate memory for matrix H[" << setw(6) << i << "]\n";
            exit(1);
        }
    }
    cout << "Memory for matrix H allocated\n\n";

    cout << "Initializing matrix H\n";
    for(int i=0; i<=N_a; i++) {
        for(int j=0; j<=N_b; j++) {
            H[i][j]=0.;
        }
    }
    cout << "Matrix H initialized\n\n";

    cout << "Allocating memory for matrix I_i\n";
    I_i = (unsigned short **)malloc((N_a + 1) * sizeof(unsigned short *));
    if (I_i == NULL) {
        cout << "Could not allocate memory for matrix I_i\n";
        exit(1);
    }
    for (int i = 0; i < (N_a + 1); i++) {
        I_i[i] = (unsigned short *)malloc((N_b + 1) * sizeof(unsigned short));
        if (I_i[i] == NULL) {
            cout << "Could not allocate memory for matrix I_i[" << setw(6) << i << "]\n";
            exit(1);
        }
    }
    cout << "Memory for matrix I_i allocated\n\n";

    cout << "Allocating memory for matrix I_j\n";
    I_j = (unsigned short **)malloc((N_a + 1) * sizeof(unsigned short *));
    if (I_j == NULL) {
        cout << "Could not allocate memory for matrix I_j\n";
        exit(1);
    }
    for (int i = 0; i < (N_a + 1); i++) {
        I_j[i] = (unsigned short *)malloc((N_b + 1) * sizeof(unsigned short));
        if (I_j[i] == NULL) {
            cout << "Could not allocate memory for matrix I_j[" << setw(6) << i << "]\n";
            exit(1);
        }
    }
    cout << "Memory for matrix I_j allocated\n\n";

	cout << "Initialize threads data\n";
	thread_data_t *td;
	td=setup_thread_data(N_threads, N_a, N_b);
	pthread_t thread[N_threads];
	pthread_barrier_init(&bar,NULL,N_threads);
	cout << "Threads initialized\n";
    // here comes the thread creation!
    gettimeofday(&StartTime, NULL);
    
/* CDF START */
    //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif
/* CDF END */    
    
	for(int i=0;i<N_threads;i++){
		pthread_create(&thread[i], NULL, p_SmithWaterman, (void*)&td[i]);
	}

	for(int j=0;j<N_threads;j++){
		pthread_join(thread[j],NULL);
	}

	pthread_barrier_destroy(&bar);

#if 0
    // Print the matrix H to the console
    cout<<"**********************************************"<<endl;
    cout<<"The scoring matrix is given by  "<<endl<<endl;
    for(int i=1; i<=N_a; i++) {
        for(int j=1; j<=N_b; j++) {
            cout<<H[i][j]<<" ";
        }
        cout<<endl;
    }
#endif

    // search H for the maximal score
	pthread_t thread2[N_threads];
	pthread_mutex_init(&mutex,NULL);

	for(int i=0;i<N_threads;i++){
		pthread_create(&thread2[i], NULL, search_max_score, (void*)&td[i]);
	}

	for(int j=0;j<N_threads;j++){
		pthread_join(thread[j],NULL);
	}
	pthread_mutex_destroy(&mutex);

    // Backtracking from H_max
    int current_i = i_max, current_j = j_max;
    int next_i = I_i[current_i][current_j];
    int next_j = I_j[current_i][current_j];
    int tick = 0;
    char consensus_a[N_a+N_b+2], consensus_b[N_a+N_b+2];

    while(((current_i!=next_i) || (current_j!=next_j)) && (next_j!=0) && (next_i!=0)) {

        if(next_i==current_i)  consensus_a[tick] = '-';                  // deletion in A
        else                   consensus_a[tick] = seq_a[current_i-1];   // match/mismatch in A

        if(next_j==current_j)  consensus_b[tick] = '-';                  // deletion in B
        else                   consensus_b[tick] = seq_b[current_j-1];   // match/mismatch in B

        current_i = next_i;
        current_j = next_j;
        next_i = I_i[current_i][current_j];
        next_j = I_j[current_i][current_j];
        tick++;
    }
    
/* CDF START */
    //  Right before the section of the code we want to measure
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif
/* CDF END */

    gettimeofday(&EndTime, NULL);

//    cout<<endl<<"***********************************************"<<endl;
//    cout<<"The alignment of the sequences"<<endl<<endl;
//    for(int i=0; i<N_a; i++) {
//        cout<<seq_a[i];
//    };
//    cout<<"  and"<<endl;
//    for(int i=0; i<N_b; i++) {
//        cout<<seq_b[i];
//    };
//    cout<<endl<<endl;
//    cout<<"is for the parameters  mu = "<<mu<<" and delta = "<<delta<<" given by"<<endl<<endl;
//    for(int i=tick-1; i>=0; i--) cout<<consensus_a[i];
//    cout<<endl;
//    for(int j=tick-1; j>=0; j--) cout<<consensus_b[j];
//    cout<<endl;
//
//    if (EndTime.tv_usec < StartTime.tv_usec) {
//        int nsec = (StartTime.tv_usec - EndTime.tv_usec) / 1000000 + 1;
//        StartTime.tv_usec -= 1000000 * nsec;
//        StartTime.tv_sec += nsec;
//    }
//    if (EndTime.tv_usec - StartTime.tv_usec > 1000000) {
//        int nsec = (EndTime.tv_usec - StartTime.tv_usec) / 1000000;
//        StartTime.tv_usec += 1000000 * nsec;
//        StartTime.tv_sec -= nsec;
//    }

	cout<<"Last Score: "<<H[N_a][N_b]<<endl;
	cout << "Max Score: "<<H_max<<endl;
//    printf("\n\nParallel calculation time: %ld.%.6ld seconds\n", EndTime.tv_sec  - StartTime.tv_sec, EndTime.tv_usec - StartTime.tv_usec);
    
/* CDF START */
    // Before you exit the application
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif
/* CDF END */
    
    return 0;

}

