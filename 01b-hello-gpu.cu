#include <cstdio>

__global__ void kernel(void) {
	printf("hello from GPU kernel: threadIdx.x=%d\n", threadIdx.x);
}

int main(void) {
	printf("begin GPU kernel\n");
	kernel <<< 1, 16>>>();
	cudaDeviceSynchronize();
	printf("end GPU kernel\n");
	fflush(stdout);
	return 0;
}

// HISTORY: hello-gpu.cu
static const char rcsid[] __attribute__((used)) = "$Id: 01b-hello-gpu.cu,v 1.2 2026/08/29 01:27:50 wayfarecru Exp $";
