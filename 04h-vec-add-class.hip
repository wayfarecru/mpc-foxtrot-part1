#include "hip/hip_runtime.h"
#include <cstdio>

#include "./common.hpp"

// CUDA kernel function
template<typename T>
__global__ void kernel_vec_add( T* C, const T* A, const T* B, int N ) {
	int i = blockIdx.x * blockDim.x + threadIdx.x; // CUDA-provided index
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}

class VecAdd {
protected:
#if defined(__NVCC__)
	const unsigned N = 1024 * 1024 * 1024; // 1G elements
#else 
	const unsigned N = 512 * 1024 * 1024; // 512M elements
#endif
	// host-side
	float* ptrA;
	float* ptrB;
	float* ptrC;
	// device-side
	float* devPtrA;
	float* devPtrB;
	float* devPtrC;
public:
	void prepare_host(void) {
		// host-side data
		ptrA = new float[N];
		ptrB = new float[N];
		ptrC = new float[N];
		// set random data
		srand( time(NULL) );
		setRand( N, ptrA );
		setRand( N, ptrB );
	}
	void copy_to_device(void) {
		// allocate device memory
		hipMalloc( (void**)&devPtrA, N * sizeof(float) );
		hipMalloc( (void**)&devPtrB, N * sizeof(float) );
		hipMalloc( (void**)&devPtrC, N * sizeof(float) );
		// host to device copy
		hipMemcpy( devPtrA, ptrA, N * sizeof(float), hipMemcpyHostToDevice );
		hipMemcpy( devPtrB, ptrB, N * sizeof(float), hipMemcpyHostToDevice );
		CUDA_CHECK_ERROR();
	}
	void execute_kernel(void) {
		// kernel launch
		kernel_vec_add<float> <<< N / 1024, 1024>>>( devPtrC, devPtrA, devPtrB, N );
		hipDeviceSynchronize();
		CUDA_CHECK_ERROR();
	}
	void copy_to_host(void) {
		// device to host copy
		hipMemcpy( ptrC, devPtrC, N * sizeof(float), hipMemcpyDeviceToHost );
		CUDA_CHECK_ERROR();
	}
	void check(void) {
		printf("N = %d\n", N);
		printf("A[%d] = ", N);
		printArray(N, ptrA);
		printf("B[%d] = ", N);
		printArray(N, ptrB);
		printf("C[%d] = ", N);
		printArray(N, ptrC);
	}
	void clear(void) {
		// free device memory
		hipFree( devPtrA );
		hipFree( devPtrB );
		hipFree( devPtrC );
		CUDA_CHECK_ERROR();
		// cleaning
		delete[] ptrA;
		delete[] ptrB;
		delete[] ptrC;
	}
};


int main(void) {
	VecAdd vecadd;
	vecadd.prepare_host();
	vecadd.copy_to_device();
	vecadd.execute_kernel();
	vecadd.copy_to_host();
	vecadd.check();
	vecadd.clear();
	// done
	return 0;
}

// HISTORY: vec-add-class.cu
static const char rcsid[] __attribute__((used)) = "$Id: 04h-vec-add-class.cu,v 1.3 2026/08/29 02:14:34 wayfarecru Exp $";
