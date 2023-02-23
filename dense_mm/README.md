# Matrix multiplication.
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


