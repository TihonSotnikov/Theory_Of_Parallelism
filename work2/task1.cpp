#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

void init_data(double*& vec, double*& matrix, double*& result, int size)
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

void matrix_vector_product(const double* vec, const double* matrix, double* result, int size)
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

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <M>\n";
        return 1;
    }
    
    int m = std::atoi(argv[1]);

    double *vec = nullptr, *matrix = nullptr, *result = nullptr;
    double T_sequential = 0.0;

    std::vector<int> nums_threads = {1, 2, 4, 6, 8, 16, 20, 40};

    std::cout << "Threads  Time(s)  Speedup\n";
    std::cout << "-------------------------\n";

    std::cout << std::fixed << std::setprecision(5);
    for (int cur_num_threads : nums_threads)
    {
        omp_set_num_threads(cur_num_threads);
        init_data(vec, matrix, result, m);

        double start_time = omp_get_wtime();
        matrix_vector_product(vec, matrix, result, m);
        double end_time = omp_get_wtime();
        double elapsed_time = end_time - start_time;
        if (cur_num_threads == 1)
            T_sequential = elapsed_time;
        double speedup = T_sequential / elapsed_time;

        std::cout << cur_num_threads << "        " 
                  << elapsed_time << "  " 
                  << speedup << '\n';

        free_data(vec, matrix, result);
    }

    return 0;
}
