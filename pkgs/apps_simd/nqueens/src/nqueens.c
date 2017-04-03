#include <stdio.h>
//#include <math.h>
#include <stdlib.h>
#include <time.h>

// The number of solutions
int solutions = 0;

// How many queens we are using
const int queens = 12;

//FILE *f;


/*void kiir(char *A)
{
    int i, j;
    
    solutions++;
    
	
    for(i = 0; i < 20; i++)
	printf("\b");
    
    printf("%d", solutions);
    
    fprintf(f, "%d\n", solutions);

    
    for(i = 0; i < queens; i++)
    {
        for(j = 0; j < queens; j++)
            if(A[i] == j)
                fprintf(f, "Q ");
            else
                fprintf(f, ". ");

        fprintf(f, "\n");
    }

    fprintf(f, "\n");
}*/

/**
 *  A: Array for storing the chessboard
 *  k: How many queens can be placed on the board
 *  n: How many queens have been placed on the board
 */
void findQueenSolutions(char *A, 
						const int k, 
						const int n)
{
	// Set last element equal to k
	A[n-1] = k;

	// Row & column variables must be 
	// stored separately to enable backtracking
	int i, j;
    
	// Loop through whole chessboard. End when tiles for 
	// x,y is the same tile, or when the distance between 
	// x,y is the same as the distance between the tiles 
	// for x,y
	for(i = 0; i < n - 1; i++)
		for(j = i + 1; j < n; j++)
			if(A[i] == A[j] || abs(i - j) == abs(A[i] - A[j]))
				return;
	
	// We can validly place the queen (last for loop).
	// IF we have placed n queens, add this as solution.
	// If not, increment queens placed and run again.
	if(n == queens)
		solutions++;
	else
		for(i = 0; i < queens; i++)
		{
			char B[n+1];
			for(j = 0; j < n; j++)
				B[j] = A[j];
			findQueenSolutions(B, i, n+1);
		}
}

// Time-keepinbg function
double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}

int main()
{
	printf("[NQUEENS] Number of queens: %d\n", queens);
	//scanf("%d", &queens);

	//f = fopen("out", "w");
    
	//printf("Number of solutions:\n");


	double timeElapsed = GetWallTime();

	for(int i = 0; i < queens; i++)
	{
		// From 0 to N Queens, create a 1x1 array containing 0
		// and run the solver
		char A[1] = {0};
		findQueenSolutions(A, i, 1);
    }

	timeElapsed = GetWallTime() - timeElapsed;

    printf("[NQUEENS] Number of solutions: %d\n", solutions);
	printf("[NQUEENS] Time used: %f seconds\n", timeElapsed);
    //printf("\n");
    //fclose(f);

    return 0;
}
