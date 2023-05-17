# 1D stencil
* 1D stencil is an operation that for any point in an element in a vector is the sum of the  same element in the input vector + Elements before that number with RADIUS number + Elements after that number with RADIUS number. 
 * The elements before 0 counted as 0
 * The elements after vector SIZE counted as 0

* Example:
As in Figure 1, vector size=10, and radius=2 
  * To calculate b[0] = a[-2]+a[-1]+a[0]+a[1]+a[2] = 0+0+10+4+5. Because a[-1], and a[-2] before location 0, counted as 0
  * To calculate b[1] = a[-1]+a[0]+a[1]+a[2]+a[3] = 0+10+4+5+6. Because a[-1] before location 0, counted as 0
  * To calculate b[4] = a[2]+a[3]+a[4]+a[5]+a[6] = 5+7+9+11+13.
  * To calculate b[9] = a[7]+a[8]+a[9]+a[10]+a[11] = 20+0+3+0+. Because a[10], and a[11] after vector size, counted as 0

![image](https://github.com/compilereg/parallel-codes/blob/main/1dstencil/example1.png)
Figure 1

# 1d stencil codes:
## Sequential code
The code, calculates the time taken by function stencil_1d by surrounding its call using function time. The function time returns current time. Using function difftime that accepts two times, calculates the time difference in seconds.
 * To compile the code, gcc -o stencil_seq stencil_seq.c 
 * To submit the code to the cluster, sbatch submit.seq "./stencil_seq 1000 10"
 * To run the code locally, ./stencil_seq 1000 10
## Parallel code with OMP
The code, calculates the time taken by function stecil_id by calling omp_get_wtime just before call it, and one time after called it. Calculates difference between two times.
 * To compile the code, gcc -o stencil_omp stencil_omp.c -fopenmp
 * To submit the code to the cluster, sbatch submit.openmp "./stencil_omp 1000 10"
 * To run the code locally, ./stencil_omp 1000 10
## Parallel code with CUDA
The calculation of stencil offloaded to the kernel function "stencil_1d_gpu". The kernel launched into grid with configuration of ceil(size/BLOCKSIZE) blocks, and BLOCKSIZE threads in each block. In each thread, the index calculated as threadIDx.x + blockDim.x * blockIdx.x. 
Keep in mind, also in each thread, the kernel function must check the vector index inside the range from 0 to size-1
 * To compile the code, submit.nvcc stencil-cuda stencil-cuda.cu
 * To submit the code to the cluster, sbatch submit.gpu "./stencil-cuda 1000 10" where 1000 is the vector suze, and 10 is the radius.
