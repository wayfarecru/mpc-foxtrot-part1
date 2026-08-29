#include <cstdlib>
#include <cstring>
#include <typeinfo>

// options for compilers

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4388 4389)
#endif

// check error macro

#if defined(__NVCC__)
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
#endif

#if defined(__HIP__)
#if defined(NDEBUG)
#define CUDA_CHECK_ERROR()	0
#else
#define CUDA_CHECK_ERROR()	do { \
        hipError_t e = hipGetLastError(); \
        if (hipSuccess != e) { \
            printf("rocm failure \"%s\" at %s:%d\n", \
                   hipGetErrorString(e), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif
#endif

// CUDA/ROCm warm up

#if defined(__NVCC__)
// GPU dummy kernel for warm-up
__global__ void __hidden_kernel_warmup(void) {
	// do nothing
}

inline void CUDA_WARM_UP(void) {
	cudaFree(0);
	__hidden_kernel_warmup <<< 1, 1>>>();
}
#endif

#if defined(__HIP__)
// GPU dummy kernel for warm-up
__global__ void __hidden_kernel_warmup(void) {
	1.0F + 1.0F; // meaningless operation
}

inline void CUDA_WARM_UP(void) {
	hipFree(0);
	__hidden_kernel_warmup <<< 1, 1>>>();
}
#endif


// print kernel configuration

#define CUDA_PRINT_CONFIG(dimx)	do { \
		printf("prob size = %d\n", dimx); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

#define CUDA_PRINT_CONFIG_2D(dimx,dimy)	do { \
		printf("prob size = %d * %d\n", dimx, dimy); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

#define CUDA_PRINT_CONFIG_3D(dimx,dimy,dimz)	do { \
		printf("prob size = %d * %d * %d\n", dimx, dimy, dimz); \
		printf("gridDim   = %d * %d * %d\n", dimGrid.x, dimGrid.y, dimGrid.z); \
		printf("blockDim  = %d * %d * %d\n", dimBlock.x, dimBlock.y, dimBlock.z); \
		printf("total thr = %d * %d * %d\n", dimGrid.x*dimBlock.x, dimGrid.y*dimBlock.y, dimGrid.z*dimBlock.z); \
		fflush(stdout); \
	} while (0)

inline void printArray(int n, const float* arr) {
	for (int i = 0; i < 3; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf(". . . ");
	for (int i = n - 3; i < n; ++i) {
		printf("%10.3f ", arr[i]);
	}
	printf("\n");
}

inline void printArray(int n, const int* arr) {
	for (int i = 0; i < 3; ++i) {
		printf("%10d ", arr[i]);
	}
	printf(". . . ");
	for (int i = n - 3; i < n; ++i) {
		printf("%10d ", arr[i]);
	}
	printf("\n");
}

inline void printMat(int nrow, int ncol, float* mat) {
	int c = ncol;
#define M(row,col) mat[(row)*ncol+(col)]
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(0, 0), M(0, 1), M(0, 2), M(0, c - 3), M(0, c - 2), M(0, c - 1));
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(1, 0), M(1, 1), M(1, 2), M(1, c - 3), M(1, c - 2), M(1, c - 1));
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(2, 0), M(2, 1), M(2, 2), M(2, c - 3), M(2, c - 2), M(2, c - 1));
	printf("\t........ ........ ........ ... ........ ........ ........\n");
	int r = nrow - 3;
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
	r = nrow - 2;
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
	r = nrow - 1;
	printf("\t%8f %8f %8f ... %8f %8f %8f\n", M(r, 0), M(r, 1), M(r, 2), M(r, c - 3), M(r, c - 2), M(r, c - 1));
#undef M
}

inline void setRand(int n, float* arr) {
	for (int i = 0; i < n; ++i) {
		arr[i] = (rand() / (float)(RAND_MAX)); // values will be [0.0, 1.0). 
	}
}

inline void setRand(int n, int* arr) {
	for (int i = 0; i < n; ++i) {
		arr[i] = rand();
	}
}

inline void setRand(int n, int* arr, int bound) {
	for (int i = 0; i < n; ++i) {
		arr[i] = (rand() % bound);
	}
}

inline int parseNum(int argc, const char* argv[], int num = 1024*1024) {
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

template <typename TYPE>
TYPE procArg( const char* progname, const char* str, TYPE lbound = -1, TYPE ubound = -1) {
	char* pEnd = nullptr;
	TYPE value = 0;
	if (typeid(TYPE) == typeid(float) && typeid(TYPE) == typeid(double)) {
		value = strtof( str, &pEnd );
	} else {
		value = strtol( str, &pEnd, 10 );
	}
	// extra suffix processing
	if (typeid(TYPE) != typeid(float) && typeid(TYPE) != typeid(double)) {
		if (pEnd != nullptr && *pEnd != '\0') {
			switch (*pEnd) {
			case 'k': case 'K': value *= 1024; break;
			case 'm': case 'M': value *= (1024 * 1024); break;
			case 'g': case 'G': value *= (1024 * 1024 * 1024); break;
			default:
				printf("%s: ERROR: illegal parameter '%s'\n", progname, str);
				exit(EXIT_FAILURE); // EINVAL: invalid argument
				break;
			}
		}
	}
	// check for bounds
	if (lbound != -1 && value < lbound) {
		printf("%s: ERROR: invalid value '%s'\n", progname, str);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	if (ubound != -1 && value > ubound) {
		printf("%s: ERROR: invalid value '%s'\n", progname, str);
		exit(EXIT_FAILURE); // EINVAL: invalid argument
	}
	// done
	return value;
}


// get RMS (root-mean-square) error
float getRMS( const float* a, const float* b, int size, bool verbose=false ) {
	float sum = 0.0f;
	int count = size;
	float max_err = 0.0f;
	int max_pos = 0;
	while (count--) {
		float err = (*a++) - (*b++);
		sum += err * err;
		if (err > max_err) {
			max_err = err;
			max_pos = (size - (count + 1));
		}
	}
	if (verbose) {
		printf("getRMS: max_err = %f, pos = %d\n", max_err, max_pos);
	}
	return sqrtf( sum / size );
}


