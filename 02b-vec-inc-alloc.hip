#include "hip/hip_runtime.h"
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

__global__ void kernel_inc(float* devPtrDst, const float* devPtrSrc, float X) {
	int i = threadIdx.x;
	devPtrDst[i] = devPtrSrc[i] + X;
}

int main(void) {
	// CPU part: set value to A
	ptrA = (float*)malloc(N * sizeof(float));
	ptrC = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		ptrA[i] = (float)i;
	}
	printArray(N, ptrA);
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrA, N * sizeof(float));
	hipMalloc((void**)&devPtrC, N * sizeof(float));
	// host to device copy
	hipMemcpy(devPtrA, ptrA, N * sizeof(float), hipMemcpyHostToDevice); // devPtrA <- ptrA
	// kernel action
	kernel_inc <<< 1, N>>>(devPtrC, devPtrA, 1.0F);
	hipDeviceSynchronize();
	// device to host copy
	hipMemcpy(ptrC, devPtrC, N * sizeof(float), hipMemcpyDeviceToHost); // ptrC <- devPtrC
	printArray(N, ptrC);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: vec-inc-alloc.cu
static const char rcsid[] __attribute__((used)) = "$Id: 02b-vec-inc-alloc.cu,v 1.2 2026/08/29 01:56:55 wayfarecru Exp $";
