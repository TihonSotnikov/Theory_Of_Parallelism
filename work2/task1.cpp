#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static void matvec(const double* A, const double* x, double* y, int M) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = 0; j < M; j++)
            sum += A[(long)i * M + j] * x[j];
        y[i] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <M> <threads>\n", argv[0]);
        return 1;
    }

    const int M        = atoi(argv[1]);
    const int threads  = atoi(argv[2]);
    omp_set_num_threads(threads);

    double* A = new double[(long)M * M];
    double* x = new double[M];
    double* y = new double[M];

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++)
            A[(long)i * M + j] = (i == j) ? 2.0 : 1.0;
        x[i] = 1.0;
    }

    double t0 = omp_get_wtime();
    matvec(A, x, y, M);
    double elapsed = omp_get_wtime() - t0;

    printf("M=%d  threads=%d  time=%.4f s\n", M, threads, elapsed);

    delete[] A;
    delete[] x;
    delete[] y;
    return 0;
}