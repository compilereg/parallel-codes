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
![image](https://github.com/compilereg/parallel-codes/blob/main/vector/n-1.png)
Figure 2
#### Vector multiplication with 1 block, and M threads : vec_mul_1knt.cu
#### Vector multiplication with N block, and M threads : vec_mul_nknt.cu
