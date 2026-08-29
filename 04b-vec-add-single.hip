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

float* ptrA; // host memory (main memory)
float* ptrB; // host memory (main memory)
float* ptrC; // host memory (main memory)

float* devPtrA; // device memory (graphics memory)
float* devPtrB; // device memory (graphics memory)
float* devPtrC; // device memory (graphics memory)

// CPU kernel function
__host__ void kernel_cpu_vec_add(int idx, float* C, const float* A, const float* B) {
	int i = idx;
	C[i] = A[i] + B[i];
}

// GPU kernel function
__global__ void kernel_vec_add(float* C, const float* A, const float* B, const int N) {
	for (int i = 0; i < N; ++i) {
		C[i] = A[i] + B[i];
	}
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	ptrA = (float*)malloc(N * sizeof(float));
	ptrB = (float*)malloc(N * sizeof(float));
	ptrC = (float*)malloc(N * sizeof(float));
	assert(ptrA != nullptr);
	assert(ptrB != nullptr);
	assert(ptrC != nullptr);
	setRand(N, ptrA);
	setRand(N, ptrB);
	printf("A[%d] = ", N);
	printArray(N, ptrA);
	printf("B[%d] = ", N);
	printArray(N, ptrB);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_vec_add(i, ptrC, ptrA, ptrB);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's C[%d] = ", N);
	printArray(N, ptrC);
	memset(ptrC, 0, N * sizeof(float)); // clear ptrC
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: memory allocation
	hipMalloc((void**)&devPtrA, N * sizeof(float));
	hipMalloc((void**)&devPtrB, N * sizeof(float));
	hipMalloc((void**)&devPtrC, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	hipMemcpy(devPtrA, ptrA, N * sizeof(float), hipMemcpyHostToDevice); // devPtrA <- ptrA
	hipMemcpy(devPtrB, ptrB, N * sizeof(float), hipMemcpyHostToDevice); // devPtrB <- ptrB
	CUDA_CHECK_ERROR();
	// kernel action
	start = steady_clock::now();
	kernel_vec_add <<< 1, 1>>>(devPtrC, devPtrA, devPtrB, N);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	hipMemcpy(ptrC, devPtrC, N * sizeof(float), hipMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's C[%d] = ", N);
	printArray(N, ptrC);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: vec-add-single.cu <-- rand-vec-add.cu <-- chrono-vec-add.cu <-- vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 04b-vec-add-single.cu,v 1.3 2026/08/29 02:14:30 wayfarecru Exp $";
