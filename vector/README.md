# Vector multiplication
Is a matehmatical operations, takes two vectors (arrays) with the same length, multiply them element by element,  stores the result into another vector with the same length.
As in Figure 1, 3 vectors with the same length a,b, and c. Multiply each element from vector a with the same element location in vector b, and stores the result in the same element location in vector c.


![image](https://github.com/compilereg/parallel-codes/blob/main/vector/vect_mul.png)
Figure 1

## Codes
### Sequential code
The code, calculates the time taken by function dotproduct by surrounding its call using function time. The function time returns current time. Using function difftime that accepts two times, calculates the time difference in seconds.
 * To compile the code, gcc -o vec_mul_seq vec_mul_seq.c
 * To submit the code to the cluster, sbatch submit.seq "./vec_mul_seq 1000000"
 * To run the code locally, ./vec_mul_seq 1000000"
### OMP Parallel code
The code, calculates the time taken by function dotproduct by calling omp_get_wtime just before call it, and one time after called it. Calculates difference between two times.
 * To compile the code, gcc -o vec_mul_omp vec_mul_omp.c -fopenmp
 * To submit the code to the cluster, sbatch submit.openmp "./vec_mul_omp 1000000"
 * To run the code locally, ./vec_mul_omp 1000000
### GPU parallel code
#### Vector multiplication with N blocks, and 1 thread : vec_mul_nk1t.cu
In this code, created N blocks equals to vector size with 1 thread each. The kernel function dotproduct will be launched N times. In each thread, program accesses the memory location pointed by BlockIdx.x which is the block number. as in Figure 2.
 * To time taken calculated as sum of ( time consumed for copying data from host memory to device memory  + time consumed for  GPU core computation + time consumed for copying data from device memory to host memory )
 * Time calculated using cudaEventElapsedTime comibined with cudaEventCreate, cudaEventRecord, and cudaEventSynchronize.
 * The function cudaEventSynchronize is very critical, the kernel launching is a synchronus, be means once launched the control go back to the CPU without waitting to finish the task running, so the time calculation will not be accurate. The function acts a barrier, so all threads will wait till completes before executing the cudaEventElapsedTime.
 * The compilation must be done in machine with nVidia GPU installed. To compile using cluster : submit.nvcc vec_mul_nk1t vec_mul_nk1t.cu
 * To submit to the cluster : sbatch submit.gpu "./vec_mul_nk1t 10000"
![image](https://github.com/compilereg/parallel-codes/blob/main/vector/n-1.png)
Figure 2

#### Vector multiplication with 1 block, and M threads : vec_mul_1knt.cu
In this code, created 1 blocks with N thread equals to vector size . The kernel function dotproduct will be launched N times. In each thread, program accesses the memory location pointed by ThreadIdx.x which is the thread number. as in Figure 3. Here, a new concept here, in the kernel function we have to check for memory location to do not access a memory location beond the memory limits. Keep in mind, when scheduling threads to a block, all block threads will be launched even if allocated threads less than actual THREADS_PER_BLOCK
 * For example, support THREADS_PER_BLOCK = 512, and size=100, means that there are 512 threads run which yields that all threads from 101 up to 512 access an invalid memory locations.
 * To prevent this, the kernel function must check which thread run by threadIdx.x
 * To time taken calculated as sum of ( time consumed for copying data from host memory to device memory  + time consumed for GPU core computation + time consumed for copying data from device memory to host memory )
 * Time calculated using cudaEventElapsedTime comibined with cudaEventCreate, cudaEventRecord, and cudaEventSynchronize.
 * The function cudaEventSynchronize is very critical, the kernel launching is a synchronus, be means once launched the control go back to the CPU without waitting to finish the task running, so the time calculation will not be accurate. The function acts a barrier, so all threads will wait till completes before executing the cudaEventElapsedTime.
 *  The compilation must be done in machine with nVidia GPU installed. To compile using cluster : submit.nvcc vec_mul_1knt vec_mul_1knt.cu
 *  To submit to the cluster : sbatch submit.gpu "./vec_mul_1knt 10000"

![image](https://github.com/compilereg/parallel-codes/blob/main/vector/1-n.png)
Figure 3

#### Vector multiplication with N block, and M threads : vec_mul_nknt.cu
In this code, created N blocks with M thread , that NxM equals vector size . The kernel function dotproduct will be launched MxN times. In each thread, program accesses the memory location pointed by threadIdx.x + blockIdx.x * blockDim.x. as in Figure 4. Here, a new concept here, in the kernel function we have to check for memory location to do not access a memory location beond the memory limits. Keep in mind, when scheduling threads to a block, all block threads will be launched even if allocated threads less than actual THREADS_PER_BLOCK
 * To prevent this, the kernel function must check which thread run by hreadIdx.x + blockIdx.x * blockDim.x
 * To time taken calculated as sum of ( time consumed for copying data from host memory to device memory  + time consumed for GPU core computation + time consumed for copying data from device memory to host memory )
 * Time calculated using cudaEventElapsedTime comibined with cudaEventCreate, cudaEventRecord, and cudaEventSynchronize.
 * The function cudaEventSynchronize is very critical, the kernel launching is a synchronus, be means once launched the control go back to the CPU without waitting to finish the task running, so the time calculation will not be accurate. The function acts a barrier, so all threads will wait till completes before executing the cudaEventElapsedTime.
 * The compilation must be done in machine with nVidia GPU installed. To compile using cluster : submit.nvcc vec_mul_1knt vec_mul_1knt.cu
 * To submit to the cluster : sbatch submit.gpu "./vec_mul_1knt 10000"

![image](https://github.com/compilereg/parallel-codes/blob/main/vector/n-n.png)
Figure 4
