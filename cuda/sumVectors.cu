#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define VECSIZE 64000
/*
	A Cuda example, generates two vectors and sum them using cuda by offloading 3 vectors (arrays) of double data type to device.
	1-Allocate three polong inters in host memory (CPU)
	2-Fill two of them with data
	3-Allocate three polong inters in device memory (GPU)
	4-Copy data generated from step 2 to the GPU memory
	5-Define the block size for the kernel, and launch the kernel
	6-Copy result from GPU back to host memory
	7-Prlong int all vectors
	8-Release host, and device memory
	To compile the code:
	nvcc -o sumVectors sumVectors.cu
	to execute :
	./sumVectors
	Note:
		1-Check for GPU utilization by opening a new terminal. In the new terminal and before execute the code, run the command: sudo nvidia-smi
		2-variable starts with host_ means stored in host memory
		3-variable starts with dev_ means stored in device memory
*/


//Choose a random value between min and max
double randFrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

//Function accepts a vector and fill it with data
void genData(double *arr) {
for(long int i=0;i<VECSIZE;i++)
	arr[i]=randFrom(-1.0,1.0);
}

//Function accepts a vector and fill it with 0
void fillZero(double *arr) {
for(long int i=0;i<VECSIZE;i++)
        arr[i]=0;
}



//Define the kernel to be executed in threads block using keyword __global__

__global__ void addVectors(double *a,double *b,double *c,long int vecsize) {

//Now, to add value from vector a, to value from vector b, and stores the result long into vector c. To do this task, we need to calculate at which index we will work. blockIdx.x will return the block number that will be used as the index of value
long int index = blockIdx.x;

//Now, we have to check that the current index is in the range from 0 to VECSIZE-1
if ( index >= 0 && index < vecsize)
	c[index] = a[index] + b[index];
}


int main() {
double *host_a,*host_b,*host_c;
double *dev_a,*dev_b,*dev_c;
//Allocate 3 polong inters in host memory
printf("\nAllocating host memory");
host_a = (double *)malloc(sizeof(double) * VECSIZE);
if ( ! host_a ) {
	printf("\nError: Can not allocate 1st polong inter in host memory\n");
	return 1;
}
host_b = (double *)malloc(sizeof(double) * VECSIZE);
if ( ! host_b ) {
        printf("\nError: Can not allocate 2nd polong inter in host memory\n");
        return 1;
}
host_c = (double *)malloc(sizeof(double) * VECSIZE);
if ( ! host_c ) {
        printf("\nError: Can not allocate 3rd polong inter in host memory\n");
        return 1;
}
//Initialize the seed
srand ( time(NULL));
//Fill arrays with random value
genData(host_a);
genData(host_b);
//Fill array with 0
fillZero(host_c);

//Allocate 3 polong inters in device memory (GPU)
printf("\nAllocating device memory");
cudaMalloc((void **)&dev_a, VECSIZE * sizeof(double));
cudaMalloc((void **)&dev_b, VECSIZE * sizeof(double));
cudaMalloc((void **)&dev_c, VECSIZE * sizeof(double));

//Transfer data from host memory to device memory
printf("\nCopying data from host to device");
cudaMemcpy(dev_a,host_a,VECSIZE * sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(dev_b,host_b,VECSIZE * sizeof(double),cudaMemcpyHostToDevice);

//Offload the kernel to GPU and execute the kernel
printf("\nStarting the kernel");
for(int j=0;j<10000;j++)
	addVectors<<<VECSIZE,1>>>(dev_a,dev_b,dev_c,VECSIZE);


//Transfer result back from device memory to host memory
printf("\nCopying data back from device to host");
cudaMemcpy(host_c,dev_c,VECSIZE * sizeof(double), cudaMemcpyDeviceToHost);

printf("\nResult: \n");
//Prlong int values
/*
for(long int i=0;i<VECSIZE;i++) {
	printf("%f + %f = %f\n",host_a[i],host_b[i],host_c[i]);
}
*/

cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
free(host_a);
free(host_b);
free(host_c);
return 0;
}
