#include <cstdio>

void cpu_kernel(int idx) {
	printf("hello from CPU kernel: idx=%d\n", idx);
}

int main(void) {
	printf("begin CPU kernel\n");
	for (int i = 0; i < 16; ++i) {
		cpu_kernel(i);
	}
	printf("end CPU kernel\n");
	fflush(stdout);
	return 0;
}

// HISTORY: hello-cpu.cpp
static const char rcsid[] __attribute__((used)) = "$Id: 01a-hello-cpu.cpp,v 1.2 2026/08/29 01:27:02 wayfarecru Exp $";
