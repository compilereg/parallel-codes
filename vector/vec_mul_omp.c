#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
float *a,*b,*c;
int size;

//Function allocates a float vector, and return its address
float *allocateVector(int size) {
	float *t;
	t=(float *)malloc(sizeof(float) * size);
	return t;
}

//Function takes a float vector, and deallocate it
float *deallocateVector(float *v) {
	free(v);
	return v;
}

//Function takes a vector and its size, fill it with values
float *fillRand(float *v,int size) {
	int i;
	for(i=0;i<size;i++)
		v[i]=i+1;
	return v;
}

//Print a vector
void printVector(float *v,int s) {
	int i;
	printf("\n");
	for(i=0;i<s;i++) 
		printf("%2.3f ",v[i]);
}

float *dotproduct(float *a,float *b,float *c,int size) {
	int i;
#pragma omp parallel for
	for(i=0;i<size;i++)
		c[i]=a[i]*b[i];
}

int main(int argc,char **argv) {
	double diff_t;
	double start_t,end_t;
	//Check for passed parameters, the program needs one parameter , the vector size
	if ( argc != 2 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	//Reads the parameters and convert them from array of characters to integer
	size=atoi(argv[1]);
	//printf("\nVector of size  = %d ", size);
	//printf("\nAllocating vectors");
	a=allocateVector(size);
	b=allocateVector(size);
	c=allocateVector(size);
	//printf("\nFIllin in vector");
	a=fillRand(a,size);
	b=fillRand(b,size);
	
	//Get the time justt before the dotproduct function
	start_t=omp_get_wtime();
	c=dotproduct(a,b,c,size);
	//Get the time just after dorproduct function
	end_t=omp_get_wtime();
	diff_t=end_t - start_t;

	printf("\ndot product for size %d took %f seconds\n",size,diff_t);
/*
	printf("\nIn");
	printVector(in,size);
	printf("\nOut");
	printVector(out,size);
	printf("\nDeallocating vectors");
*/
	a=deallocateVector(a);
	b=deallocateVector(b);
	c=deallocateVector(c);
	return 0;
}
