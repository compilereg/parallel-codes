#include <stdio.h>
/*
	Cuda program queries total numumber of connected devices, as well as information for each device
	to compile the code:
	nvcc -o getDeviceInfo  getDeviceInfo.cu
	to exeecute the code:
	./getDeviceInfo

*/
int main() {
cudaDeviceProp prop;	//A Structure holds device information
int count,i;
cudaGetDeviceCount(&count);	//Queries total number of devices
printf("\nYou have %d devices\n",count);
for(i=0;i<count;i++) {
	//Query device infomation, and store them in the "prop" structure
	cudaGetDeviceProperties(&prop,i);
	printf("\n --- General information for device %d --- \n",i);
	printf("Name: %s\n",prop.name);
	printf("Compute capability: %d.%d\n",prop.major,prop.minor);
	printf("Clock rate: %d\n",prop.clockRate);
	printf("Device copy overlap: ");
	if ( prop.deviceOverlap )
		printf("Enabled \n");
	else
		printf("Disabled \n");
	printf("Kernel execution timeout : ");
	if ( prop.kernelExecTimeoutEnabled )
		printf("Enabled\n");
	else
		printf("Disabled\n");
	printf(" --- Memory information for device # %d : %s ---\n",i,prop.name);
	printf("Total global memory: %ld\n",prop.totalGlobalMem);
	printf("Total constant memory: %ld\n",prop.totalConstMem);
	printf("Max memory pitch: %ld\n",prop.memPitch);
	printf("Texture alignment: %ld\n",prop.textureAlignment);
	printf(" --- MP information for device # %d : %s ---\n",i,prop.name);
	printf("Multiprocessor count: %d\n",prop.multiProcessorCount);
	printf("Shared memory per MP : %ld\n",prop.sharedMemPerBlock);
	printf("Registers per MP : %d\n",prop.regsPerBlock);
	printf("Warp size : %d\n",prop.warpSize);
	printf("Max threads per block : %d\n",prop.maxThreadsPerBlock);
	printf("Max threads dimensions : %d,%d,%d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
	printf("Max grid dimensions : %d,%d,%d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
	printf("---- End for for device # %d : %s ---\n",i,prop.name);

	}
return 0;
}
