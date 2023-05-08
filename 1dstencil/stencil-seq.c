#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
float *in,*out;
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

//Function accepts an input float vector in, output float vector out, size, and radius. The function applies 1D stencil to 1D array, and store the results in the output vector
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

//Print a vector
void printVector(float *v,int s) {
	int i;
	printf("\n");
	for(i=0;i<s;i++) 
		printf("%2.3f ",v[i]);
}

int main(int argc,char **argv) {
	double diff_t;
	time_t start_t,end_t;
	//Check for passed parameters, the program needs two parameters in order, the vector size, and radius
	if ( argc != 3 ) {
		printf("\nNot enough parameters");
		return 1;
	}
	//Reads the parameters and convert them from array of characters to integer
	size=atoi(argv[1]);
	radius=atoi(argv[2]);
	//printf("\nVector of size  = %d ", size);
	//printf("\nAllocating vectors");
	in=allocateVector(size);
	out=allocateVector(size);
	//printf("\nFIllin in vector");
	in=fillRand(in,size);
	//printf("\nApply 1D Stencil to 1D array");
	
	//Get the time justt before the stencil function
	time(&start_t);
	out=stencil_1d(in,out,size,radius);
	//Get the time just after stencil function
	time(&end_t);
	diff_t = difftime(end_t,start_t);

	printf("\nStencil for size %d with radius %d took %f seconds\n",size,radius,diff_t);
/*
	printf("\nIn");
	printVector(in,size);
	printf("\nOut");
	printVector(out,size);
	printf("\nDeallocating vectors");
*/
	in=deallocateVector(in);
	out=deallocateVector(out);
	return 0;
}
