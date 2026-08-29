#include <cstdio>
#include <cstdlib>

int parseNum(int argc, char* argv[], int num = 1024 * 1024) {
	if (argc >= 2) { // get the argv[1] to long int
		char* ptr = NULL;
		long val = strtol(argv[1], &ptr, 0);
		if (ptr != NULL && *ptr != '\0') {
			switch (*ptr) {
			case 'k': case 'K': val *= 1024; break; // kilo
			case 'm': case 'M': val *= (1024 * 1024); break; // mega
			case 'g': case 'G': val *= (1024 * 1024 * 1024); break; // giga
			}
		}
		if (val > 0) {
			return val;
		}
	}
	return num;
}

int main(int argc, char* argv[], char* envp[]) {
	int N = parseNum(argc, argv, 1024);
	printf("N = %d\n", N);
	printf("argc = %d\n", argc);
	for (int i = 0; i < argc; ++i) {
		printf("argv[%d] = \"%s\"\n", i, argv[i]);
	}
	for (int i = 0; envp[i] != nullptr; ++i) {
		printf("envp[%d] = \"%s\"\n", i, envp[i]);
	}
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: parse-argc.cpp
static const char rcsid[] __attribute__((used)) = "$Id: 03i-parse-argc.cpp,v 1.3 2026/08/29 02:49:00 wayfarecru Exp $";
