#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>


#define MAX 1000
double **ma,**mb,**mc;

double ** allocateMatrix(double *m[MAX]) {
	int i;
	m=(double **)malloc(sizeof(double *)*MAX);
	for(i=0;i<MAX;i++) 
		m[i]=(double *)malloc(sizeof(double)*MAX);
	return m;
}

double ** deallocateMatrix(double *m[MAX]) {
	int i;
	for(i=0;i<MAX;i++)
		free(m[i]);
	free(m);
	return m;
}

double randfrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void initMatrix(double *m[MAX]) {
	int i,j;
	for(i=0;i<MAX;i++)
		for(j=0;j<MAX;j++)
			m[i][j]=randfrom(-1.0, 1.0);
}

void mm_update(double *a[MAX],double *b[MAX],double *c[MAX]) {
	int i,j,k;
	for(j=0;j<MAX;j++)
		for(i=0;i<MAX;i++)
			for(k=0;k<MAX;k++)
				c[i][j]+=a[i][k]*b[k][j];
}

int main() {
	time_t start_t, end_t;
	double diff_t;
	printf("\nAllocte matrices");
	ma=allocateMatrix(ma);
	mb=allocateMatrix(mb);
	mc=allocateMatrix(mc);
	printf("\nInit matrices");
	initMatrix(ma);
	initMatrix(mb);
	printf("\nStart matrix multiplication\n");
	time(&start_t);
	mm_update(ma,mb,mc);
	time(&end_t);
	diff_t = difftime(end_t, start_t);
	printf("\nTime = %f\n",diff_t);
	printf("\nDeallocating matrices");
	ma=deallocateMatrix(ma);
	mb=deallocateMatrix(mb);
	mc=deallocateMatrix(mc);


	return 0;
}
