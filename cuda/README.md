# cuda codes.
## Cuda parallel model
![image](https://github.com/compilereg/parallel-codes/blob/main/cuda/cuda-parallel.png)  
* Number of blocks called _Grid dimensions_ (field maxGridSize in property structure or blockDim.x in the kernel)
* Number of threads in the block called _Thread dimension_ (field maxThreadsDim in property structure)
* Simultaneously running threads inside the block in the same time called _warp_. Number of threads in warp called _warp size_ (warpSize in property structure)
* Current running block is __blockIdx.x__ in the kernel
* Current thread is __threadIdx.x__ in the kernel

## getDeviceInfo.cu
A cuda code that queries number of connected nvidia devices to the system. After that, query each device to print the properties.  
The program uses _cudaDeviceProp_ to retrieve the property structure for each device.  


## sumVectors.cu
A cuda code create 3 vectors (arrays), initialize 1st 2 vectors with random data, send them to the cuda device, performs the vector addition using N-block with 1 thread each.   
` long int index = blockIdx.x;
  if ( index >= 0 && index < vecsize)
	c[index] = a[index] + b[index];`  
Here, assign the kernel to only one thread in each block by getting the block number as the array index. Here every thread can not communicate with the other threads because they are running in different blocks.
