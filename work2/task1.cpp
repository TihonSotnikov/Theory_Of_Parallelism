#include <omp.h>

void matrix_vector_product_task1(const double* mat, const double* vec, double* res, size_t n)
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
