#include <cstdio>
#include <iostream>
#include <chrono>
#include <thread> // for sleep_for()
using namespace std;
using namespace std::chrono;

int main(void) {
	printf("start ---\n");
	fflush(stdout);
	steady_clock::time_point start = steady_clock::now();
	this_thread::sleep_for(milliseconds(1000));
	steady_clock::time_point end = steady_clock::now();
	printf("end ---\n");
	fflush(stdout);
	long long elapsed_msec = duration_cast<milliseconds>(end - start).count();
	cout << "elapsed time = " << elapsed_msec << " msec" << endl;
#if 0 // if you prefer C printf, use it
	printf("elapsed time = %lld msec\n", elapsed_msec);
#endif
	// done
	fflush(stdout);
	return 0;
}

// HISTORY: chrono.cpp
static const char rcsid[] __attribute__((used)) = "$Id: 03e-chrono.cpp,v 1.2 2026/08/29 02:07:38 wayfarecru Exp $";
