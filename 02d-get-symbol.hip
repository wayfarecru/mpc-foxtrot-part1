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
	hipError_t err;
	if ((err = hipGetSymbolAddress(&ptrA,HIP_SYMBOL(A))) != hipSuccess) {
		printf("hipGetSymbolAddress() failed: error code = %d\n", err);
	}
	if ((err = hipGetSymbolSize(&sizeA,HIP_SYMBOL(A))) != hipSuccess) {
		printf("hipGetSymbolSize() failed: error code = %d\n", err);
	}
	printf("A : addr=%p, size=%zu\n", devA, sizeA);
	// CPU part: set value to A
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
	// device to host copy
	// changed from: cudaMemcpyFromSymbol(C, devC, sizeof(C));
	hipMemcpy(C, devPtrC, sizeDevC, hipMemcpyDeviceToHost);
	printArray(N, C);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: get-symbol.cu <-- vec-inc-static.cu
static const char rcsid[] __attribute__((used)) = "$Id: 02d-get-symbol.cu,v 1.2 2026/08/29 02:02:32 wayfarecru Exp $";
