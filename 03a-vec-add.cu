#include <cstdio>
#include <cstring> // for memset()

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
	// CPU part: set values
	ptrA = (float*)malloc(N * sizeof(float));
	ptrB = (float*)malloc(N * sizeof(float));
	ptrC = (float*)malloc(N * sizeof(float));
	for (int i = 0; i < N; ++i) {
		ptrA[i] = (float)i;
		ptrB[i] = (float)(i * 10);
	}
	printf("A[%d] = ", N);
	printArray(N, ptrA);
	printf("B[%d] = ", N);
	printArray(N, ptrB);
	// CPU kernel
	for (int i = 0; i < N; ++i) {
		kernel_cpu_vec_add(i, ptrC, ptrA, ptrB);
	}
	printf("CPU's C[%d] = ", N);
	printArray(N, ptrC);
	memset(ptrC, 0, N * sizeof(float)); // clear ptrC
	// GPU part: memory allocation
	cudaMalloc((void**)&devPtrA, N * sizeof(float));
	cudaMalloc((void**)&devPtrB, N * sizeof(float));
	cudaMalloc((void**)&devPtrC, N * sizeof(float));
	// host to device copy
	cudaMemcpy(devPtrA, ptrA, N * sizeof(float), cudaMemcpyHostToDevice); // devPtrA <- ptrA
	cudaMemcpy(devPtrB, ptrB, N * sizeof(float), cudaMemcpyHostToDevice); // devPtrB <- ptrB
	// kernel action
	kernel_vec_add <<< 1, N>>>(devPtrC, devPtrA, devPtrB);
	cudaDeviceSynchronize();
	// device to host copy
	cudaMemcpy(ptrC, devPtrC, N * sizeof(float), cudaMemcpyDeviceToHost); // ptrC <- devPtrC
	printf("GPU's C[%d] = ", N);
	printArray(N, ptrC);
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: vec-add.cu
static const char rcsid[] __attribute__((used)) = "$Id: 03a-vec-add.cu,v 1.2 2026/08/29 02:07:33 wayfarecru Exp $";
