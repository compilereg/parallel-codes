# 1D stencil
* 1D stencil is an operation that for any point in an element in an vector is the sum of the  same element in the input vector + Elements before that number with RADIUS number + Elements after that number with RADIUS number. 
** The elements before 0 counted as 0
** The elements after vector SIZE counted as 0

* Example:
As in Figure 1, 

![image](https://github.com/compilereg/parallel-codes/blob/main/1dstencil/example1.png)
Figure 2
 
## mm_normal.c : Normal matrix multiplication
* The sequential matrix multiplication. C code, uses double pointers to allocate space for 3 matrices, ma, mb, and mc. MAX is a constant specify the matrix size.
* The program generates a random double values 
* calculates time differences before and after calling the matrix multiplication function
* To compile the code
** gcc -o mm_normal mm_normal.c
* compile the code with -O2 optimization level
** gcc -O2 -o mm_normal mm_normal.c
* Check the difference in time between difference binary generation methods

## mm_update.c : normal matrix multiplication row major
* The same as normal, but access the matrix in row major
* To compile the code
** gcc -o mm_update mm_update.c
* compile the code with -O2 optimization level
** gcc -O2 -o mm_update mm_update.c
* Check the difference in time between difference binary generation methods

## mm-update-openmp.c: parallel updated matrix multiplication in openmp
The parallelized updated matrix multiplication  in openmp
* To compile the code, gcc -O2 -o mm-update-openmp mm-update-openmp.c -fopenmp
* Check the difference in time between the 3 implementations!

## blocked_mm.c: Sequential matrix mulitplication using blocked mm algorithm


