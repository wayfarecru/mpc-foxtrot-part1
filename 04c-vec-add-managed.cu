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

float* managedPtrA; // using unified memory system
float* managedPtrB; // using unified memory system
float* managedPtrC; // using unified memory system

// CPU kernel function
__host__ void kernel_cpu_vec_add(int idx, float* C, const float* A, const float* B) {
	int i = idx;
	C[i] = A[i] + B[i];
}

// GPU kernel function
__global__ void kernel_vec_add(float* C, const float* A, const float* B, const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// allocate managed memory
	cudaMallocManaged((void**)&managedPtrA, N * sizeof(float));
	cudaMallocManaged((void**)&managedPtrB, N * sizeof(float));
	cudaMallocManaged((void**)&managedPtrC, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// give hints, if needed
#if defined(__NVCC__)
	cudaMemLocation loc = { cudaMemLocationTypeDevice, 0 };
	cudaMemAdvise(managedPtrA, N * sizeof(float), cudaMemAdviseSetPreferredLocation, loc);
	cudaMemAdvise(managedPtrB, N * sizeof(float), cudaMemAdviseSetPreferredLocation, loc);
	cudaMemAdvise(managedPtrC, N * sizeof(float), cudaMemAdviseSetPreferredLocation, loc);
#endif
#if defined(__HIP__)
	int devId = 0; // GPU id. hipMemAdvise works only for LINUX at this time.
	hipMemAdvise(managedPtrA, N * sizeof(float), hipMemAdviseSetPreferredLocation, devId);
	hipMemAdvise(managedPtrB, N * sizeof(float), hipMemAdviseSetPreferredLocation, devId);
	hipMemAdvise(managedPtrC, N * sizeof(float), hipMemAdviseSetPreferredLocation, devId);
#endif
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	setRand(N, managedPtrA);
	setRand(N, managedPtrB);
	printf("done\n");
	printf("A[%d] = ", N);
	printArray(N, managedPtrA);
	printf("B[%d] = ", N);
	printArray(N, managedPtrB);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_vec_add(i, managedPtrC, managedPtrA, managedPtrB);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	printf("CPU's C[%d] = ", N);
	printArray(N, managedPtrC);
	memset(managedPtrC, 0, N * sizeof(float)); // clear ptrC
	// GPU part: warm up
	CUDA_WARM_UP();
	// GPU part: kernel action
	start = steady_clock::now();
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernel_vec_add <<< dimGrid, dimBlock>>>(managedPtrC, managedPtrA, managedPtrB, N);
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// print out
	printf("CPU's C[%d] = ", N);
	printArray(N, managedPtrC);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: vec-add-managed.cu <-- vec-add-big.cu <-- rand-vec-add.cu <-- chrono-vec-add.cu <-- vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 04c-vec-add-managed.cu,v 1.5 2026/09/05 08:49:13 wayfarecru Exp $";
