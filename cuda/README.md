# cuda codes.
## getDeviceInfo.cu
A cuda code that queries number of connected nvidia devices to the system. After that, query each device to print the properties.  
The program uses _cudaDeviceProp_ to retrieve the property structure for each device.  

## sumVectors.cu
A cuda code create 3 vectors (arrays), initialize 1st 2 vectors with random data, send them to the cuda device, performs the vector addition using N-block with 1 thread each.
