#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
float *in,*out;
int size,radius;

float *allocateVector(int size) {
	float *t;
	t=(float *)malloc(sizeof(float) * size);
	return t;
}

float *deallocateVector(float *v) {
	free(v);
	return v;
}

float *fillRand(float *v,int size) {
	int i;
	for(i=0;i<size;i++)
		v[i]=i+1;
	return v;
}

float *stencil_1d(float *in,float *out,int size,int r) {

	int i,j;
#pragma omp parallel for private(j)
	for(i=0;i<size;i++) {
		out[i]=0;
		for(j=i-r;j<=i+r;j++) {
			if  ( j < 0 || j > size )
				continue;
			else
				out[i]+=in[j];
		}
	}
	return out;
}

void printVector(float *v,int s) {
	int i;
	printf("\n");
	for(i=0;i<s;i++) 
		printf("%2.3f ",v[i]);
}

int main(int argc,char **argv) {
	double start,end;
	if ( argc != 3 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	size=atoi(argv[1]);
	radius=atoi(argv[2]);
	//printf("\nVector of size  = %d ", size);
	//printf("\nAllocating vectors");
	in=allocateVector(size);
	out=allocateVector(size);
	//printf("\nFIllin in vector");
	in=fillRand(in,size);
	//printf("\nApply 1D Stencil to 1D array");

	start=omp_get_wtime();
	out=stencil_1d(in,out,size,radius);
	end=omp_get_wtime();

	printf("\nStencil for size %d with radius %d took %f seconds\n",size,radius,end-start);
/*
	printf("\nIn");
	printVector(in,size);
	printf("\nOut");
	printVector(out,size);
	printf("\nDeallocating vectors");
*/
	in=deallocateVector(in);
	out=deallocateVector(out);
	return 0;
}
