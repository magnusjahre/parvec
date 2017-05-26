#ifndef FUNCTIONS_H
#define FUNCTIONS_H

float  similarity_score(char a, char b);
array_max_t find_array_max(float array[],int length);
void   checkfile(int open, char filename[]);
string read_sequence(ifstream& f);

#endif
