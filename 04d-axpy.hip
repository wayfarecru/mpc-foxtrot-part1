#include "hip/hip_runtime.h"
#include <cstdio>
#include <chrono>
#include <cassert>
using namespace std;
using namespace std::chrono;

#include "./common.hpp"

#if defined(__NVCC__)
int N = 1024 * 1024 * 1024; // 1G elements: may be updated with parseNum()
#else // HIPCC
int N = 512 * 1024 * 1024; // 512M elements: may be updated with parseNum()
#endif

float a = 0.0F;
float* ptrX; // host memory (main memory)
float* ptrY; // host memory (main memory)
float* ptrZ; // host memory (main memory)

float* devPtrX; // device memory (graphics memory)
float* devPtrY; // device memory (graphics memory)
float* devPtrZ; // device memory (graphics memory)

// CPU kernel function
__host__ void kernel_cpu_axpy(int idx, float* Z, const float a, const float* X, const float* Y) {
	int i = idx;
	Z[i] = a * X[i] + Y[i];
}

// GPU kernel function
__global__ void kernel_axpy(float* Z, const float a, const float* X, const float* Y, const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		Z[i] = a * X[i] + Y[i];
	}
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	ptrX = (float*)malloc(N * sizeof(float));
	ptrY = (float*)malloc(N * sizeof(float));
	ptrZ = (float*)malloc(N * sizeof(float));
	assert(ptrX != nullptr);
	assert(ptrY != nullptr);
	assert(ptrZ != nullptr);
	setRand(1, &a);
	setRand(N, ptrX);
	setRand(N, ptrY);
	printf("A = %f\n", a);
	printf("X[%d] = ", N);
	printArray(N, ptrX);
	printf("Y[%d] = ", N);
	printArray(N, ptrY);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_axpy(i, ptrZ, a, ptrX, ptrY);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's Z[%d] = ", N);
	printArray(N, ptrZ);
	memset(ptrZ, 0, N * sizeof(float)); // clear ptrZ
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrX, N * sizeof(float));
	hipMalloc((void**)&devPtrY, N * sizeof(float));
	hipMalloc((void**)&devPtrZ, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrX, ptrX, N * sizeof(float), hipMemcpyHostToDevice); // devPtrA <- ptrA
	hipMemcpy(devPtrY, ptrY, N * sizeof(float), hipMemcpyHostToDevice); // devPtrB <- ptrB
	CUDA_CHECK_ERROR();
	// kernel action
	start = steady_clock::now();
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernel_axpy <<< dimGrid, dimBlock>>>(devPtrZ, a, devPtrX, devPtrY, N);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	hipMemcpy(ptrZ, devPtrZ, N * sizeof(float), hipMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's Z[%d] = ", N);
	printArray(N, ptrZ);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: axpy.cu <-- vec-add-big.cu <-- rand-vec-add.cu <-- chrono-vec-add.cu <-- vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 04d-axpy.cu,v 1.3 2026/08/29 02:14:31 wayfarecru Exp $";
