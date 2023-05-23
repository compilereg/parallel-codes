/*
   Calculates the stencil on GPU without sharing, all threads access the global memory
   */

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#define BLOCKSIZE 512
//out1 used to store the results back from GPU
float *in,*out,*out1;
float *d_in,*d_out;
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

//Function accepts an input float vector in, output float vector out, size, and radius. The function applies 1D stencil to 1D array, and store the results in the output vector using CPU
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

//Function accepts an input float vector in, output float vector out, size, and radius. The function applies 1D stencil to 1D array, and store the results in the output vector using GPU
__global__ void stencil_1d_gpu(float *in,float *out,int r,int size) {
    __shared__ float temp[BLOCKSIZE + 2 * r];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + r;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < r)
        {
        temp[lindex – r] = in[gindex – r];
        temp[lindex + BLOCKSIZE] = in[gindex + BLOCKSIZE];
        }
    // Synchronize (ensure all the data is available)
    __syncthreads();

       // Apply the stencil
    int result = 0;
    for (int offset = -r ; offset <= r ; offset++)
        result += temp[lindex + offset];

    // Store the result
    out[gindex] = result;
}

void checkResults(float *h_out,float *d_out,int size) {
	int i=0;
	while( h_out[i] == d_out[i] && i < size)
		i++;
	if ( i >= size)
		printf("\nCheck results: Identical");
	else
		printf("\nCheck results: Not identical at %d",i);
	printf("\n");
}


//Print a vector
void printVector(float *v,int s) {
	int i;
	printf("\n");
	for(i=0;i<s;i++) 
		printf("%2.3f ",v[i]);
}

int main(int argc,char **argv) {
	float diff_t;
	cudaEvent_t start, stop;
	int GridSize;
	time_t start_t,end_t;
	//Check for passed parameters, the program needs two parameters in order, the vector size, and radius
	if ( argc != 3 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	//Reads the parameters and convert them from array of characters to integer
	size=atoi(argv[1]);
	radius=atoi(argv[2]);
printf("\nVector of size  = %d, and radius = %d", size,radius);
	//printf("\nAllocating vectors");
	in=allocateVector(size);
	out=allocateVector(size);
	out1=allocateVector(size);
//printf("\nFIllin in vector");
	in=fillRand(in,size);
//printf("\nStart gpu");
	//Start stencil using GPU
	//create a start event
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//Allocating vectors in GPU memory
	cudaMalloc((void **)&d_in,size * sizeof(float));
	cudaMalloc((void **)&d_out,size * sizeof(float));

	//Copy in vector from host memory to device memory
	cudaMemcpy(d_in,in,size * sizeof(float) , cudaMemcpyHostToDevice);
	GridSize = ceil((float)size/BLOCKSIZE);
//printf("\nGrid size = %d",GridSize);
	stencil_1d_gpu<<<GridSize,BLOCKSIZE>>>(d_in,d_out,radius,BLOCKSIZE);

	//Copy out vector from device memory to host memory
	cudaMemcpy(out1,d_out,size * sizeof(float) , cudaMemcpyDeviceToHost);

	//Synchronize to give GPU change to complete
	cudaThreadSynchronize();

	//create a stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//Calculate time differences
	cudaEventElapsedTime(&diff_t, start, stop);
printf("\nTime taken in GPU is %f in seconds",(float)diff_t/1000);

	//Calculates stencil on CPU
	time(&start_t);
        out=stencil_1d(in,out,size,radius);
	time(&end_t);
printf("\nTime taken in CPU is %f seconds",difftime(end_t,start_t));
	//Check results between GPU and Host
	checkResults(out1,out,size);
/*
printf("\nInput : ");
printVector(in,size);
printf("\nCPU : ");
printVector(out,size);
printf("\nGPU : ");
printVector(out1,size);
printf("\n");
*/
	cudaFree(d_in);
	cudaFree(d_out);
	in=deallocateVector(in);
	out=deallocateVector(out);
	out1=deallocateVector(out1);
	return 0;
}
