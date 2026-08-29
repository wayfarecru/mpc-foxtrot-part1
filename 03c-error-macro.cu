#include <cstdio>

#if defined(NDEBUG)
#define CUDA_CHECK_ERROR()	0
#else
#define CUDA_CHECK_ERROR()	do { \
        cudaError_t e = cudaGetLastError(); \
        if (cudaSuccess != e) { \
            printf("cuda failure \"%s\" at %s:%d\n", \
                   cudaGetErrorString(e), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif

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
	cudaGetSymbolAddress(&devPtrA, devA);
	CUDA_CHECK_ERROR();
	cudaGetSymbolSize(&sizeDevA, devA);
	CUDA_CHECK_ERROR();
	printf("devA : addr=%p, size=%zu\n", devPtrA, sizeDevA);
	void* devPtrC = nullptr;
	size_t sizeDevC = 0;
	cudaGetSymbolAddress(&devPtrC, devC);
	CUDA_CHECK_ERROR();
	cudaGetSymbolSize(&sizeDevC, devC);
	CUDA_CHECK_ERROR();
	printf("devC : addr=%p, size=%zu\n", devPtrC, sizeDevC);
	// error case with a main memory symbol
	void* ptrA = nullptr;
	size_t sizeA = 0;
	cudaGetSymbolAddress(&ptrA, A);
	CUDA_CHECK_ERROR();
	cudaGetSymbolSize(&sizeA, A);
	CUDA_CHECK_ERROR();
	printf("A : addr=%p, size=%zu\n", devA, sizeA);
	// CPU part: set values
	for (int i = 0; i < N; ++i) {
		A[i] = (float)i;
	}
	printArray(N, A);
	// host to device copy
	// changed from: cudaMemcpyToSymbol(devA, A, sizeof(A));
	cudaMemcpy(devPtrA, A, sizeDevA, cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	// GPU part: kernel action
	float X = 1.0F;
	cudaMemcpyToSymbol(devX, &X, sizeof(float));
	kernel_inc <<< 1, N>>>();
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR();
	// device to host copy
	// changed from: cudaMemcpyFromSymbol(C, devC, sizeof(C));
	cudaMemcpy(C, devPtrC, sizeDevC, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();
	printArray(N, C);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: error-macro.cu <-- error-check.cu <-- get-symbol.cu <-- vec-inc-static.cu
static const char rcsid[] __attribute__((used)) = "$Id: 03c-error-macro.cu,v 1.2 2026/08/29 02:07:36 wayfarecru Exp $";
