#include <omp.h>
#include <cmath>

#ifdef schedule_dynamic
    #define s schedule(dynamic)
#elif schedule_guided
    #define s schedule(guided)
#else
    #define s schedule(static)
#endif

const double eps = 1e-5;

void find_solution_v1(const double* A, const double* b, double* x, size_t n)
{
    double* r = new double[n];
    double* Ax = new double[n];
    
    double b_norm = 0.0;
    #pragma omp parallel for reduction(+ : b_norm) s
    for (size_t i = 0; i < n; i++)
        b_norm += b[i] * b[i];
    b_norm = sqrt(b_norm);

    double tau = 1.0 / (n + 1.0);

    for (size_t iter = 0; iter < 10000; iter++)
    {
        #pragma omp parallel for s
        for (size_t i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < n; j++)
                sum += A[i * n + j] * x[j];
            Ax[i] = sum;
        }

        double norm_sq = 0.0;
        #pragma omp parallel for reduction(+ : norm_sq) s
        for (size_t i = 0; i < n; i++)
        {
            r[i] = Ax[i] - b[i];
            norm_sq += r[i] * r[i];
        }

        if (sqrt(norm_sq) / b_norm < eps)
            break;

        #pragma omp parallel for s
        for (size_t i = 0; i < n; i++)
            x[i] -= tau * r[i];
    }

    delete[] r;
    delete[] Ax;
}

void find_solution_v2(const double* A, const double* b, double* x, size_t n)
{
    double* r = new double[n];
    double* Ax = new double[n];
    
    double b_norm = 0.0;
    for (size_t i = 0; i < n; i++)
        b_norm += b[i] * b[i];
    b_norm = sqrt(b_norm);

    double tau = 1.0 / (n + 1.0);
    double norm_sq = 0.0;

    #pragma omp parallel
    {
        for (size_t iter = 0; iter < 10000; iter++)
        {
            #pragma omp for s
            for (size_t i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (size_t j = 0; j < n; j++)
                    sum += A[i * n + j] * x[j];
                Ax[i] = sum;
            }

            #pragma omp single
            norm_sq = 0.0;

            #pragma omp for reduction(+ : norm_sq) s
            for (size_t i = 0; i < n; i++)
            {
                r[i] = Ax[i] - b[i];
                norm_sq += r[i] * r[i];
            }

            if (sqrt(norm_sq) / b_norm < eps)
                break;

            #pragma omp for s
            for (size_t i = 0; i < n; i++)
                x[i] -= tau * r[i];
        }
    }

    delete[] r;
    delete[] Ax;
}
