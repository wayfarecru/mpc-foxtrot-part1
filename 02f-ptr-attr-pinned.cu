#include <cstdio>

const int N = 1024;

float* ptrA; // host memory (main memory)
float* ptrC; // host memory (main memory)

float* devPtrA; // device memory (graphics memory)
float* devPtrC; // device memory (graphics memory)

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
	case cudaMemoryTypeUnregistered:	typeStr = "unregistered"; break;
	case cudaMemoryTypeHost:	typeStr = "host"; break;
	case cudaMemoryTypeDevice:	typeStr = "device"; break;
	case cudaMemoryTypeManaged:	typeStr = "managed"; break;
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
	// CPU part: set values
	cudaMallocHost((void**)&ptrA, N * sizeof(float));
	cudaMallocHost((void**)&ptrC, N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		ptrA[i] = (float)i;
	}
	printArray(N, ptrA);
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrA, N * sizeof(float));
	cudaMalloc((void**)&devPtrC, N * sizeof(float));
	// host to device copy
	cudaMemcpy(devPtrA, ptrA, N * sizeof(float), cudaMemcpyHostToDevice); // devPtrA <- ptrA
	// kernel action
	kernel_inc <<< 1, N>>>(devPtrC, devPtrA, 1.0F);
	cudaDeviceSynchronize();
	// device to host copy
	cudaMemcpy(ptrC, devPtrC, N * sizeof(float), cudaMemcpyDeviceToHost); // ptrC <- devPtrC
	printArray(N, ptrC);
	// show pointer attributes
	cudaPointerAttributes attr;
	printf("ptrA : ");
	cudaPointerGetAttributes(&attr, ptrA);
	printPtrAttr(attr);
	printf("devPtrA : ");
	cudaPointerGetAttributes(&attr, devPtrA);
	printPtrAttr(attr);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: ptr-attr-pinned.cu <-- vec-inc-malloc-host.cu <-- vec-inc-alloc.cu
static const char rcsid[] __attribute__((used)) = "$Id: 02f-ptr-attr-pinned.cu,v 1.2 2026/08/29 02:02:49 wayfarecru Exp $";
