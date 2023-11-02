#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <pthread.h>

double **a,**b,**c;
#define NTHREADS 8
int size;

double **allocateMatrix(int s) {

	int i;
	double **a;
	a=(double **)malloc(s*sizeof(double *));
	for(i=0;i<s;i++)
		a[i]=(double *)malloc(s*sizeof(double));
	return a;
}

double **deallocateMatrix(double **a,int s) {
	int i;
	for(i=0;i<s;i++)
		free(a[i]);
	free(a);
	return a;
}


void initMatrix(double **a) {
	int i,j;
	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
			a[i][j]=1.0;
}


void initMatrix0(double **a) {
        int i,j;
        for(i=0;i<size;i++)
                for(j=0;j<size;j++)
                        a[i][j]=0;
}

/*
  void mm(double **a,double **b,double **c) {
	int i,j,k;

	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
			for(k=0;k<size;k++)
				c[i][j]+=a[i][k]*b[k][j];
}
*/

void *mm(void *t)
{
int tid;
int parsize,start,end;
int i,j,k;
tid=*(int *)t;
parsize = size / NTHREADS;
start= tid * parsize;
end=(tid+1) * parsize-1;
 for(i=start;i<=end;i++)
                for(j=0;j<size;j++)
                        for(k=0;k<size;k++)
                                c[i][j]+=a[i][k]*b[k][j];


}


int main(int argc,char *argv[]) {
	int i;
	if ( argc != 2 ) {
		printf("\nInvalid parameters");
		return 1;
	}
	pthread_t p[NTHREADS];

	size=atoi(argv[1]);
	printf("\nSize=%d",size);
	a=allocateMatrix(size);
	if ( ! a ) printf("\nCan not allocate matrix a");
	b=allocateMatrix(size);
	if ( ! b ) printf("\nCan not allocate matrix b");
	c=allocateMatrix(size);
	if ( ! c ) printf("\nCan not allocate matrix c");
	initMatrix(a);
	initMatrix(b);
	initMatrix0(c);

	
	for(i=0;i<NTHREADS;i++) {
		pthread_create(&p[i],NULL,mm,(void *)&i);
	}

	for(i=0;i<NTHREADS;i++)
		pthread_join(p[i],NULL);

	a=deallocateMatrix(a,size);
	b=deallocateMatrix(b,size);
	c=deallocateMatrix(c,size);
	return 0;
}
