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

float t = 0.5F;
float* ptrX; // host memory (main memory)
float* ptrY; // host memory (main memory)
float* ptrZ; // host memory (main memory)

float* devPtrX; // device memory (graphics memory)
float* devPtrY; // device memory (graphics memory)
float* devPtrZ; // device memory (graphics memory)

// CPU kernel function
__host__ void kernel_cpu_lerp(int idx, float* Z, const float t, const float* X, const float* Y) {
	int i = idx;
	Z[i] = (1.0F - t) * X[i] + t * Y[i];
}

// GPU kernel function
__global__ void kernel_lerp(float* Z, const float t, const float* X, const float* Y, const int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		Z[i] = (1.0F - t) * X[i] + t * Y[i];
	}
}

int main(int argc, const char* argv[]) {
	srand(time(NULL));
	N = parseNum(argc, argv, N);
	printf("N = %d\n", N);
	// CPU part: set values
	printf("generating %d random numbers: ", N);
	fflush(stdout);
	ptrX = (float*)malloc(N * sizeof(float));
	ptrY = (float*)malloc(N * sizeof(float));
	ptrZ = (float*)malloc(N * sizeof(float));
	assert(ptrX != nullptr);
	assert(ptrY != nullptr);
	assert(ptrZ != nullptr);
	setRand(N, ptrX);
	setRand(N, ptrY);
	printf("done\n");
	t = 0.5f;
	printf("t = %f\n", t);
	printf("X[%d] = ", N);
	printArray(N, ptrX);
	printf("Y[%d] = ", N);
	printArray(N, ptrY);
	fflush(stdout);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_lerp(i, ptrZ, t, ptrX, ptrY);
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
	cudaMalloc((void**)&devPtrX, N * sizeof(float));
	cudaMalloc((void**)&devPtrY, N * sizeof(float));
	cudaMalloc((void**)&devPtrZ, N * sizeof(float));
	CUDA_CHECK_ERROR();
	// host to device copy
	cudaMemcpy(devPtrX, ptrX, N * sizeof(float), cudaMemcpyHostToDevice); // devPtrA <- ptrA
	cudaMemcpy(devPtrY, ptrY, N * sizeof(float), cudaMemcpyHostToDevice); // devPtrB <- ptrB
	CUDA_CHECK_ERROR();
	// kernel action
	start = steady_clock::now();
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernel_lerp <<< dimGrid, dimBlock>>>(devPtrZ, t, devPtrX, devPtrY, N);
	cudaDeviceSynchronize();
	end = steady_clock::now();
	elapsed_usec = duration_cast<microseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %.3f msec\n", elapsed_usec / 1000.0F);
	CUDA_CHECK_ERROR();
	// device to host copy
	cudaMemcpy(ptrZ, devPtrZ, N * sizeof(float), cudaMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's Z[%d] = ", N);
	printArray(N, ptrZ);
	CUDA_CHECK_ERROR();
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: lerp.cu <-- axpy.cu <-- vec-add-big.cu <-- rand-vec-add.cu <-- chrono-vec-add.cu <-- vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 04f-lerp.cu,v 1.4 2026/08/29 02:48:38 wayfarecru Exp $";
