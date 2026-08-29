#include <cstdio>

__global__ void kernel(void) {
	printf("hello from GPU kernel: threadIdx.x=%d\n", threadIdx.x);
}

int main(void) {
	printf("begin GPU kernel\n");
	kernel <<< 4, 4>>>();
	cudaDeviceSynchronize();
	printf("end GPU kernel\n");
	fflush(stdout);
	return 0;
}

// HISTORY: hello-gpu44.cu <-- hello.gpu.cu
static const char rcsid[] __attribute__((used)) = "$Id: 01c-hello-gpu44.cu,v 1.2 2026/08/29 01:28:22 wayfarecru Exp $";
