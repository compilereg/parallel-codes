#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc,char **argv) {
	int vecLength;
	int partSize;
	int rank,size;
	if ( argc != 2 ) {
		printf("\nInvalid args");
		return 1;
	}

	vecLength=atoi(argv[1]);

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if ( vecLength % (size-1) != 0 )
	{
		printf("\nVector size must be divisible by cluster size");
		return 2;
	}
	partSize = vecLength / (size-1);
	//Master node
	if ( rank == 0 ) {
		int i;
		double *v1,*v2,*v3;
		v1=(double *)malloc(sizeof(double)*vecLength);
		v2=(double *)malloc(sizeof(double)*vecLength);
		v3=(double *)malloc(sizeof(double)*vecLength);
		for(i=1;i<vecLength;i++) {
			v1[i]=v2[i]=i;
			v3[i]=0;
		}
		for(i=1;i<size;i++)
			MPI_Send(&partSize,1,MPI_INT,i,0,MPI_COMM_WORLD);
		for(i=1;i<size;i++){
			MPI_Send(v1+partSize*(i-1),partSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
			MPI_Send(v2+partSize*(i-1),partSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD);
		}
		for(i=1;i<size;i++)
			MPI_Recv(v3+partSize*(i-1),partSize,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		for(i=0;i<vecLength;i++)
			printf("\n %f x %f = %f\n",v1[i],v2[i],v3[i]);
		free(v1);
		free(v2);
		free(v3);
	} else {	
	int nodePartSize;
	double *nodev1,*nodev2,*nodev3;
	MPI_Recv(&nodePartSize,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	nodev1=(double *)malloc(sizeof(double)*nodePartSize);
	nodev2=(double *)malloc(sizeof(double)*nodePartSize);
	nodev3=(double *)malloc(sizeof(double)*nodePartSize);
	MPI_Recv(nodev1,nodePartSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(nodev2,nodePartSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	for(int j=0;j<nodePartSize;j++)
		nodev3[j] = nodev1[j] * nodev2[j];
	MPI_Send(nodev3,nodePartSize,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
	free(nodev1);
	free(nodev2);
	free(nodev3);
	}
	MPI_Finalize();
	return 0;
}
