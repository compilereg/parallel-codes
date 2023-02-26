#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>


//#define MAX 500
int MAX,blocksize;
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

void mm_blocked(double *a[MAX],double *b[MAX],double *c[MAX],int bs) {
	int i,j,k;
	int ii,jj,kk;
	for(ii=0;ii<MAX;ii+=bs)
		for(jj=0;jj<MAX;jj+=bs)
			for(kk=0;kk<MAX;kk+=bs)
				for(i=ii;i<(ii+bs);i++)
					for(j=jj;j<(jj+bs);j++)
						for(k=kk;k<(kk+bs);k++)
							c[i][j]+=a[i][k]*b[k][j];
}

int main(int argc,char *argv[]) {
	time_t start_t, end_t;
	double diff_t;
	if ( argc != 3 ) {
		fprintf(stderr,"Insufficient parameters, send matrix size, followed by block size");
		return 1;
	}
	MAX=atoi(argv[1]);
	blocksize=atoi(argv[2]);
	printf("\nAllocte matrices");
	ma=allocateMatrix(ma);
	mb=allocateMatrix(mb);
	mc=allocateMatrix(mc);
	printf("\nInit matrices");
	initMatrix(ma);
	initMatrix(mb);
	printf("\nStart matrix multiplication\n");
	time(&start_t);
	mm_blocked(ma,mb,mc,blocksize);
	time(&end_t);
	diff_t = difftime(end_t, start_t);
	printf("\nTime = %f\n",diff_t);
	printf("\nDeallocating matrices");
	ma=deallocateMatrix(ma);
	mb=deallocateMatrix(mb);
	mc=deallocateMatrix(mc);
	return 0;
}
