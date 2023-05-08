/*
	A CUDA code, initalizes 2 float vectors, and multiply them using GPU, stores the result in 3rd vector.
	The operations done on N Kernal, 1 thread each
*/
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
float *a,*b,*c;
float *da,*db,*dc;
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

//The code performs vector multiplication, in each block 
__global__ void dotproduct(float *da,float *db,float *dc) {
	/*
		blockIdx.x : block number
	*/
	int loc=blockIdx.x;
	dc[loc] = da[loc] * db[loc];
}

int main(int argc,char **argv) {
	//commStart, commStop used to record time 
	cudaEvent_t commStart,commStop;
	float comptime;
	//Check for passed parameters, the program needs one parameter , the vector size
	if ( argc != 2 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	//Reads the parameters and convert them from array of characters to integer
	size=atoi(argv[1]);
	//printf("\nVector of size  = %d ", size);
	//printf("\nAllocating vectors");
	//Allocation 3 vectors in host memory
	a=allocateVector(size);
	b=allocateVector(size);
	c=allocateVector(size);

	//Allocation of 3 vectors in device memory
	cudaMalloc((void **)&da,size*sizeof(float));
	cudaMalloc((void **)&db,size*sizeof(float));
	cudaMalloc((void **)&dc,size*sizeof(float));

	//Filling 2 host vectors
	//printf("\nFIllin in vector");
	a=fillRand(a,size);
	b=fillRand(b,size);

	cudaEventCreate(&commStart);
	cudaEventCreate(&commStop);
	cudaEventRecord(commStart,0);

	//Copy content of vector a, and b to the vectors stored in GPU memory
	cudaMemcpy(da,a,size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(db,b,size*sizeof(float),cudaMemcpyHostToDevice);


	//Launch the kernel with blocks = size, and block size =1
	dotproduct<<<size,1>>>(da,db,dc);


	//Copy back result from GPU memory vector dc, to the host memory vector c

	cudaMemcpy(c,dc,size*sizeof(float),cudaMemcpyDeviceToHost);

	cudaEventRecord(commStop,0);

        cudaEventSynchronize(commStop);
        cudaEventElapsedTime(&comptime,commStart,commStop);
	comptime=comptime/1000;
	
	printf("\ndot product for size %d took %f seconds\n",size,comptime);

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
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	return 0;
}
