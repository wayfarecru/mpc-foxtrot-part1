#include <cstdio>

const int N = 1024;

float* managedPtrA; // using unified memory system
float* managedPtrC; // using unified memory system

void printArray(int n, const float* arr) {
	for (int i = 0; i < 4; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf(". . . ");
	for (int i = n - 4; i < n; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf("\n");
}

void printPtrAttr(const cudaPointerAttributes& attr) {
	char const* typeStr = "undefined";
	switch (attr.type) {
	case cudaMemoryTypeUnregistered:    typeStr = "unregistered"; break;
	case cudaMemoryTypeHost:    typeStr = "host"; break;
	case cudaMemoryTypeDevice:  typeStr = "device"; break;
	case cudaMemoryTypeManaged: typeStr = "managed"; break;
#if defined(__HIP__)
	case hipMemoryTypeArray:	typeStr = "array"; break;
	case hipMemoryTypeUnified:	typeStr = "unified"; break;
#endif
	}
	printf("type=%s, device=%d, devPtr=%p, hostPtr=%p\n",
	       typeStr, attr.device, attr.devicePointer, attr.hostPointer);
}


__global__ void kernel_inc(float* devPtrDst, const float* devPtrSrc, float X) {
	int i = threadIdx.x;
	devPtrDst[i] = devPtrSrc[i] + X;
}

int main(void) {
	// allocate managed memory
	cudaError_t err;
	if ((err = cudaMallocManaged((void**)&managedPtrA, N * sizeof(float))) != cudaSuccess) {
		printf("cudaMallocManaged failed: error code = %d\n", err);
		exit(0);
	}
	if ((err = cudaMallocManaged((void**)&managedPtrC, N * sizeof(float))) != cudaSuccess) {
		printf("cudaMallocManaged failed: error code = %d\n", err);
		exit(0);
	}
	// give hints, if needed
#if defined(__NVCC__)
	cudaMemLocation loc = { cudaMemLocationTypeDevice, 0 };
	cudaMemAdvise(managedPtrA, N * sizeof(float), cudaMemAdviseSetPreferredLocation, loc);
	cudaMemAdvise(managedPtrC, N * sizeof(float), cudaMemAdviseSetPreferredLocation, loc);
#endif
#if defined(__HIP__)
	int devId = 0; // GPU id. hipMemAdvise works only for LINUX at this time.
	hipMemAdvise(managedPtrA, N * sizeof(float), hipMemAdviseSetPreferredLocation, devId);
	hipMemAdvise(managedPtrC, N * sizeof(float), hipMemAdviseSetPreferredLocation, devId);
#endif
	// CPU part: set value to A
	for (int i = 0; i < N; ++i) {
		managedPtrA[i] = (float)i;
	}
	printArray(N, managedPtrA);
	// kernel action
	kernel_inc <<< 1, N>>>(managedPtrC, managedPtrA, 1.0F);
	cudaDeviceSynchronize();
	// print result
	printArray(N, managedPtrC);
	// show pointer attributes
	cudaPointerAttributes attr;
	printf("managedPtrA : ");
	cudaPointerGetAttributes(&attr, managedPtrA);
	printPtrAttr(attr);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: managed-mem.cu <-- ptr-attr.cu <-- vec-inc-alloc.cu
static const char rcsid[] __attribute__((used)) = "$Id: 02g-managed-mem.cu,v 1.2 2026/08/29 02:02:58 wayfarecru Exp $";
