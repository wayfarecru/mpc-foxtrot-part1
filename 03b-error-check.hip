#include "hip/hip_runtime.h"
#include <cstdio>

const int N = 1024; // problem size

float A[N]; // host memory (main memory)
float C[N]; // host memory (main memory)

void printArray(int n, const float arr[]) {
	for (int i = 0; i < 4; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf(". . . ");
	for (int i = n - 4; i < n; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf("\n");
}

__device__ float devA[N]; // device memory (graphics memory)
__device__ float devC[N]; // device memory (graphics memory)
__device__ float devX = 0.0F;

__global__ void kernel_inc(void) {
	int i = threadIdx.x; // index
	devC[i] = devA[i] + devX;
}

int main(void) {
	// get symbol address and size
	void* devPtrA = nullptr;
	size_t sizeDevA = 0;
	hipGetSymbolAddress(&devPtrA,HIP_SYMBOL(devA));
	hipGetSymbolSize(&sizeDevA,HIP_SYMBOL(devA));
	printf("devA : addr=%p, size=%zu\n", devPtrA, sizeDevA);
	void* devPtrC = nullptr;
	size_t sizeDevC = 0;
	hipGetSymbolAddress(&devPtrC,HIP_SYMBOL(devC));
	hipGetSymbolSize(&sizeDevC,HIP_SYMBOL(devC));
	printf("devC : addr=%p, size=%zu\n", devPtrC, sizeDevC);
	// error case with a main memory symbol
	void* ptrA = nullptr;
	size_t sizeA = 0;
	if (hipGetSymbolAddress(&ptrA,HIP_SYMBOL(A)) != hipSuccess) {
		hipError_t err = hipGetLastError();
		printf("hipGetSymbolAddress() failed: %s (code=%d)\n", hipGetErrorString(err), err);
	}
	if (hipGetSymbolSize(&sizeA,HIP_SYMBOL(A)) != hipSuccess) {
		hipError_t err = hipGetLastError();
		printf("hipGetSymbolSize() failed: %s (code=%d)\n", hipGetErrorString(err), err);
	}
	printf("A : addr=%p, size=%zu\n", devA, sizeA);
	// CPU part: set values
	for (int i = 0; i < N; ++i) {
		A[i] = (float)i;
	}
	printArray(N, A);
	// host to device copy
	// changed from: cudaMemcpyToSymbol(devA, A, sizeof(A));
	hipMemcpy(devPtrA, A, sizeDevA, hipMemcpyHostToDevice);
	// GPU part: kernel action
	float X = 1.0F;
	hipMemcpyToSymbol(HIP_SYMBOL(devX), &X, sizeof(float));
	kernel_inc <<< 1, N>>>();
	hipDeviceSynchronize();
	if (hipPeekAtLastError() != hipSuccess) {
		hipError_t err = hipGetLastError();
		printf("GPU kernel failed: %s (code=%d)\n", hipGetErrorString(err), err);
	} else {
		hipError_t err = hipGetLastError();
		printf("GPU kernel success: %s (code=%d)\n", hipGetErrorString(err), err);
	}
	// device to host copy
	// changed from: cudaMemcpyFromSymbol(C, devC, sizeof(C));
	hipMemcpy(C, devPtrC, sizeDevC, hipMemcpyDeviceToHost);
	printArray(N, C);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: error-check.cu <-- get-symbol.cu <-- vec-inc-static.cu
static const char rcsid[] __attribute__((used)) = "$Id: 03b-error-check.cu,v 1.2 2026/08/29 02:07:34 wayfarecru Exp $";
