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
