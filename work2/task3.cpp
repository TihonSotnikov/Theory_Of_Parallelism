#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static int solve_v1(const double* A, double* x, const double* b, int N, double tau, double eps) {
    double* r = new double[N];
    int iter = 0;
    while (true) {
        double norm_r = 0.0, norm_b = 0.0;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double s = -b[i];
            for (int j = 0; j < N; j++) s += A[(long)i * N + j] * x[j];
            r[i] = s;
        }

        #pragma omp parallel for schedule(static) reduction(+:norm_r, norm_b)
        for (int i = 0; i < N; i++) {
            norm_r += r[i] * r[i];
            norm_b += b[i] * b[i];
        }

        if (sqrt(norm_r) / sqrt(norm_b) < eps) break;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            x[i] -= tau * r[i];

        iter++;
    }
    delete[] r;
    return iter;
}

static int solve_v2(const double* A, double* x, const double* b, int N, double tau, double eps) {
    double* r = new double[N];
    int iter = 0;

    #pragma omp parallel
    {
        while (true) {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                double s = -b[i];
                for (int j = 0; j < N; j++) s += A[(long)i * N + j] * x[j];
                r[i] = s;
            }

            double norm_r = 0.0, norm_b = 0.0;
            #pragma omp for schedule(static) reduction(+:norm_r, norm_b)
            for (int i = 0; i < N; i++) {
                norm_r += r[i] * r[i];
                norm_b += b[i] * b[i];
            }

            #pragma omp single
            { iter++; }

            if (sqrt(norm_r) / sqrt(norm_b) < eps) break;

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++)
                x[i] -= tau * r[i];
        }
    }

    delete[] r;
    return iter;
}

static void init(double* A, double* x, double* b, int N) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            A[(long)i * N + j] = (i == j) ? 2.0 : 1.0;
        b[i] = N + 1.0;
        x[i] = 0.0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <N> <threads>\n", argv[0]);
        return 1;
    }

    const int N       = atoi(argv[1]);
    const int threads = atoi(argv[2]);
    const double tau  = 1.0 / (2.0 * N);
    const double eps  = 1e-5;
    omp_set_num_threads(threads);

    double* A = new double[(long)N * N];
    double* b = new double[N];
    double* x = new double[N];

    init(A, b, x, N);

    double t0 = omp_get_wtime();
    int iters = solve_v1(A, x, b, N, tau, eps);
    double t1 = omp_get_wtime() - t0;
    printf("v1  N=%d  threads=%d  iters=%d  time=%.4f s\n", N, threads, iters, t1);

    // reset x
    #pragma omp parallel for
    for (int i = 0; i < N; i++) x[i] = 0.0;

    t0 = omp_get_wtime();
    iters = solve_v2(A, x, b, N, tau, eps);
    double t2 = omp_get_wtime() - t0;
    printf("v2  N=%d  threads=%d  iters=%d  time=%.4f s\n", N, threads, iters, t2);

    delete[] A;
    delete[] b;
    delete[] x;
    return 0;
}