#include <omp.h>
#include <cmath>

const double eps = 1e-8;

void matrix_vector_product(const double* mat, const double* vec, double* res, size_t n)
{
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++)
            sum += mat[i * n + j] * vec[j];
        res[i] = sum;
    }
}

double vector_norm(const double* vec, size_t n)
{
    double norm = 0.0;

    #pragma omp parallel for reduction(+ : norm)
    for (size_t i = 0; i < n; i++)
        norm += vec[i] * vec[i];

    return sqrt(norm);
}

double vector_dot_product(const double* vec1, const double* vec2, size_t n)
{
    double res = 0.0;
    #pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; i++)
        res += vec1[i] * vec2[i];

    return res;
}

void add_scaled_vector(double* vec1, const double* vec2, double scal, size_t n)
{
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        vec1[i] += vec2[i] * scal;
}

void find_solution(const double* A, const double* b, double* x, size_t n)
{
    double* r = new double[n];
    double* Ax = new double[n];
    double* w = new double[n];
    double b_norm = vector_norm(b, n);

    for (size_t iter = 0; iter < 10000; iter++)
    {
        matrix_vector_product(A, x, Ax, n);

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            r[i] = b[i] - Ax[i];

        if (vector_norm(r, n) / b_norm < eps)
            break;

        matrix_vector_product(A, r, w, n);

        double tau = vector_dot_product(w, r, n) / vector_dot_product(w, w, n);

        add_scaled_vector(x, r, tau, n);
    }

    delete[] r;
    delete[] Ax;
    delete[] w;
}
