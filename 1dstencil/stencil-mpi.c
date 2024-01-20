/*
 * Calculate 1d stencil in MPI
 */
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <mpi.h>
#include <string.h>

#define MASTER 0
float *in,*out,*outmpi,*inv,*ouv;
int size,radius;

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

//Function accepts an input float vector in, output float vector out, size, and radius. The function applies 1D stencil to 1D array, and store the results in the output vector
float *stencil_1d(float *in,float *out,int size,int r) {

	int i,j;
	//Loop on the output vector from the beginning to the end
	for(i=0;i<size;i++) {
		//Initialize the current output element 
		out[i]=0;
		/*
		Loop on the input vector to load the elements starting from the current elemnt - radius to current element + radius
		All elements in locations smaller than 0 counted as 0
		All elements in locations greater than vector size counted as 0
		*/	
		for(j=i-r;j<=i+r;j++) {
			//Check if the input location less than 0, or greater than vector size, skip the current iteration
			if  ( j < 0 || j > size )
				continue;
			else
				out[i]+=in[j];
		}
	}
	return out;
}

//Print a vector
void printVector(float *v,int s) {
	int i;
	printf("\n");
	for(i=0;i<s;i++) 
		printf("%2.3f ",v[i]);
}

int main(int argc,char **argv) {
	double diff_t;
	time_t start_t,end_t;
	int nProc,myRank,i;
	int partSize,index,partStart,partEnd;
	int nodePartSize;
        int nodeRadius;

	//Check for passed parameters, the program needs two parameters in order, the vector size, and radius
	if ( argc != 3 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	//Reads the parameters and convert them from array of characters to integer
	size=atoi(argv[1]);
	radius=atoi(argv[2]);
//Initialize MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nProc);
	MPI_Comm_rank(MPI_COMM_WORLD,&myRank);
/*
        If the node is the master node
            1)allocate memory
            2)fill inut vector
            3)parition the data:
                Here, we can send all input vector to all compute nodes. It is a simple, but there will be huge unneeded data sent to compute nodes.
                Instead, we will send to each node the exact data needed, with elements after and before with radiud to avoid
                    communication between nodes during calculation
                    1-Calculate 1st element
                    2-Calculate the last element
                    3-Total size of elements
            4)Snd the data
            5)recieve the results
*/
	if ( myRank == MASTER ) {
		//Allocate memory for master input vector,
		in=allocateVector(size);
    		//Allocate memory for out vector calculated sequentially
		out=allocateVector(size);
    		//Allocate memory for out vector calculated by MPI
		outmpi=allocateVector(size);
    		//Fill the master input vector with random values
		in=fillRand(in,size);
  /*
    	Start partitioning starting node #0 which is the master node. Master node will be used as compute node also.
    		To exclude the master node from compution,
        1-Change loop index from i=0 to i=1
        2-Change partSize=size/nProc to partSize=size/(nProc-1);

  */
		partSize=size/nProc;
		for ( i=0 ; i < nProc;i++) {
            /*
                There are 3 cases to calculate the start, end, and part size for each compute node
                1-Start-Radius before 0
                2-Start+Radius after size::
                3-Start-Radius && Start+Radius inside vector boundaries
                */
                partStart=i*partSize-radius;
                partEnd=i*partSize + partSize + radius-1;
            	if ( partStart < 0 )   {
                	partStart=0;
            	}
            	if ( partEnd > size ) {
                	partEnd=size-1;
            	}
            	nodePartSize=partEnd-partStart + 1;
		//Start sending each partition to the compute node including the master node
		//Here, the master node also will compute its own portion of the vector
		MPI_Send(&nodePartSize,1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Send(&radius,1,MPI_INT,i,0,MPI_COMM_WORLD);
		MPI_Send(in+partStart,nodePartSize,MPI_FLOAT,i,0,MPI_COMM_WORLD);
		}
	}
	//Compute node (including master ) to calculate stencil for its partition
	MPI_Recv(&nodePartSize,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(&nodeRadius,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	inv=allocateVector(nodePartSize);
	ouv=allocateVector(nodePartSize);
	//MPI_Recv(inv,nodePartSize+2*nodeRadius,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	MPI_Recv(inv,nodePartSize,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	//Verify data received at each node by printing them
	char line[300],line1[300],line2[300];
	sprintf(line,"%d receives Part size=%d, and radius=%d , data=",myRank,nodePartSize,radius);
	for(int v=0;v<nodePartSize;v++) {
		sprintf(line2,"[%d]=%f : ",v,inv[v]);
		strcat(line,line2);
	}
	printf("\n%s\n",line);
	
	//Calculate the 1D stencil in each compute node

	//Start receiving results
	if ( myRank == MASTER) {
		//HEre, the master should receive the values in outmpi vector
	}
	//Here, all compute node should send its out values to the master node
	//
	//Deallocatin vectors , and finalizing MPI
	inv=deallocateVector(inv);
        ouv=deallocateVector(ouv);


	//Calculate 1d stencil in sequential, and compare the results
	MPI_Finalize();

	if ( myRank == MASTER) {
	//Get the time justt before the stencil function
	//time(&start_t);
	out=stencil_1d(in,out,size,radius);
	i=0;
	while( i<size ) {
		if ( outmpi[i] != out[i]) {
			printf("\nWrong value at %d",i);
			break;
		}
		else 
			i++;
	}
	if ( i >= size )
		printf("\nEvery thing is ok");
	//Get the time just after stencil function
	//time(&end_t);
	//diff_t = difftime(end_t,start_t);

	//printf("\nStencil for size %d with radius %d took %f seconds\n",size,radius,diff_t);
/*
	printf("\nIn");
	printVector(in,size);
	printf("\nOut seq");
	printVector(out,size);
	printf("\nDeallocating vectors");
*/
	//Deallocating all vectors
	in=deallocateVector(in);
	out=deallocateVector(out);
	outmpi=deallocateVector(outmpi);
	}
	return 0;
}
