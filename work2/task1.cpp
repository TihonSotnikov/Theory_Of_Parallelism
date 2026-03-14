#include <omp.h>

void init_data(double*& vec, double*& matrix, double*& result, const int size)
{
    vec = new double[size];
    matrix = new double[(long long)size * size];
    result = new double[size];

    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        vec[i] = i;
        result[i] = 0.0;
        for (int j = 0; j < size; j++)
            matrix[(long long)i * size + j] = i + j;
    }
}

void free_data(double* vec, double* matrix, double* result)
{
    delete[] vec;
    delete[] matrix;
    delete[] result;
}

void matrix_vector_product(const double* vec, const double* matrix, double* result, const int size)
{
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < size; j++)
            sum += matrix[(long long)i * size + j] * vec[j];
        result[i] = sum;
    }
}
