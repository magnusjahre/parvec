#include "globals.h"
#include "functions.h"

/******************************************************************************/
/* auxiliary functions used by main                                           */
/******************************************************************************/

void checkfile(int open, char filename[]) {

    if (open) {
        cout << "Error: Can't open the file "<<filename<<endl;
        exit(1);
    }
    else cout<<"Opened file \"" << filename << "\"\n";
}

/******************************************************************************/

float similarity_score(char a,char b) {

    float result;
    if(a==b) {
        result=1.;
    }
    else {
        result=-mu;
    }
    return result;
}

/******************************************************************************/

array_max_t find_array_max(float array[],int length) {

	array_max_t a_m;
	a_m.max = array[0]; 			// start with max = first element
	a_m.ind=0;
    for(int i = 1; i<length; i++) {
        if(array[i] > a_m.max) {
            a_m.max = array[i];
            a_m.ind = i;
        }
    }
    return a_m;                    // return highest value in array
}

/******************************************************************************/

string read_sequence(ifstream& f)
{
    string seq;
    char line[20000];
    while( f.good() )
    {
        f.getline(line,20000);
        if( line[0] == 0 || line[0]=='#' )
            continue;
        for(int i = 0; line[i] != 0; ++i)
        {
            int c = toupper(line[i]);
            if( c != 'A' && c != 'G' && c != 'C' && c != 'T' )
                continue;

            seq.push_back(char(c));
        }
    }
    return seq;
}

/******************************************************************************/
