#include "hip/hip_runtime.h"
#include <cstdio>
#include <cstring> // for memset()
#include <chrono>
using namespace std;
using namespace std::chrono;

#if defined(NDEBUG)
#define CUDA_CHECK_ERROR()	0
#else
#define CUDA_CHECK_ERROR()	do { \
        hipError_t e = hipGetLastError(); \
        if (hipSuccess != e) { \
            printf("cuda failure \"%s\" at %s:%d\n", \
                   hipGetErrorString(e), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif

const int N = 1024;

float* ptrA; // host memory (main memory)
float* ptrB; // host memory (main memory)
float* ptrC; // host memory (main memory)

float* devPtrA; // device memory (graphics memory)
float* devPtrB; // device memory (graphics memory)
float* devPtrC; // device memory (graphics memory)

void printArray(int n, const float* arr) {
	for (int i = 0; i < 3; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf(". . . ");
	for (int i = n - 3; i < n; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf("\n");
}

void setRand(int n, float* arr) {
	for (int i = 0; i < n; ++i) {
		arr[i] = (rand() / (float)(RAND_MAX)); // values will be [0.0, 1.0). 
	}
}

// CPU kernel function
__host__ void kernel_cpu_vec_add(int idx, float* C, const float* A, const float* B) {
	int i = idx;
	C[i] = A[i] + B[i];
}

// GPU kernel function
__global__ void kernel_vec_add(float* C, const float* A, const float* B) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

int main(void) {
	srand(time(NULL));
	// CPU part: set values
	ptrA = (float*)malloc(N * sizeof(float));
	ptrB = (float*)malloc(N * sizeof(float));
	ptrC = (float*)malloc(N * sizeof(float));
	setRand(N, ptrA);
	setRand(N, ptrB);
	printf("A[%d] = ", N);
	printArray(N, ptrA);
	printf("B[%d] = ", N);
	printArray(N, ptrB);
	// CPU kernel
	steady_clock::time_point start = steady_clock::now();
	for (int i = 0; i < N; ++i) {
		kernel_cpu_vec_add(i, ptrC, ptrA, ptrB);
	}
	steady_clock::time_point end = steady_clock::now();
	long long elapsed_msec = duration_cast<milliseconds>(end - start).count();
	printf("CPU kernel: elapsed time = %lld msec\n", elapsed_msec);
	printf("CPU's C[%d] = ", N);
	printArray(N, ptrC);
	memset(ptrC, 0, N * sizeof(float)); // clear ptrC
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
	kernel_vec_add <<< 1, N>>>(devPtrC, devPtrA, devPtrB);
	hipDeviceSynchronize();
	end = steady_clock::now();
	elapsed_msec = duration_cast<milliseconds>(end - start).count();
	printf("GPU kernel: elapsed time = %lld msec\n", elapsed_msec);
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

// HISTORY: rand-vec-add.cu <-- chrono-vec-add.cu <-- vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 03h-rand-vec-add.cu,v 1.2 2026/08/29 02:07:43 wayfarecru Exp $";
