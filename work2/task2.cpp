#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static double integrate_atomic(int nsteps) {
    double dx = 1.0 / nsteps;
    double sum = 0.0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nsteps; i++) {
        double x = (i + 0.5) * dx;
        double val = 4.0 / (1.0 + x * x);
        #pragma omp atomic
        sum += val;
    }
    return sum * dx;
}

static double integrate_local(int nsteps) {
    double dx = 1.0 / nsteps;
    double sum = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (int i = 0; i < nsteps; i++) {
        double x = (i + 0.5) * dx;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum * dx;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <threads>\n", argv[0]);
        return 1;
    }

    const int nsteps  = 40000000;
    const int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    double t0, elapsed, result;

    t0 = omp_get_wtime();
    result = integrate_atomic(nsteps);
    elapsed = omp_get_wtime() - t0;
    printf("atomic  threads=%d  pi=%.10f  time=%.4f s\n", threads, result, elapsed);

    t0 = omp_get_wtime();
    result = integrate_local(nsteps);
    elapsed = omp_get_wtime() - t0;
    printf("local   threads=%d  pi=%.10f  time=%.4f s\n", threads, result, elapsed);

    return 0;
}