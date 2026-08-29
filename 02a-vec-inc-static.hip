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

__global__ void kernel_inc(const float X) {
	int i = threadIdx.x; // index
	devC[i] = devA[i] + X;
}

int main(void) {
	// CPU part: set value to A
	for (int i = 0; i < N; ++i) {
		A[i] = (float)i;
	}
	printArray(N, A);
	// host to device copy
	hipMemcpyToSymbol(HIP_SYMBOL(devA), A, sizeof(A));
	// GPU part: kernel action
	kernel_inc <<< 1, N>>>(1.0);
	hipDeviceSynchronize();
	// device to host copy
	hipMemcpyFromSymbol(C,HIP_SYMBOL(devC), sizeof(C));
	printArray(N, C);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: vec-inc-static.cu
static const char rcsid[] __attribute__((used)) = "$Id: 02a-vec-inc-static.cu,v 1.2 2026/08/29 01:56:31 wayfarecru Exp $";
