#include <cstdio>
#include <cstdlib> // for rand()
#include <ctime> // for time()

int main(void) {
	printf("RAND_MAX = %d\n", RAND_MAX);
	// seed 0 test
	srand(0);
	printf("random numbers, with seed 0\n");
	for (int i = 0; i < 10; ++i) {
		printf("%d\n", rand());
	}
	// histogram test
	srand(time(NULL)); // different seed for each execution
	enum { N = 1000000, M = 20 };
	printf("histogram test with %d trials into A[%d]\n", N, M);
	int A[M] = { 0 };
	for (int i = 0; i < N; ++i) {
		int ind = (rand() % M);
		A[ind]++;
	}
	for (int i = 0; i < M; ++i) {
		printf("A[%2d] = %8d\n", i, A[i]);
	}
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: rand.cpp
static const char rcsid[] __attribute__((used)) = "$Id: 03g-rand.cpp,v 1.2 2026/08/29 02:07:41 wayfarecru Exp $";
