#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstddef>

#include "task1.cpp"
#include "task2.cpp"
#include "task3.cpp"

void free_data(double* a, double* b, double* c)
{
    delete[] a;
    delete[] b;
    delete[] c;
}

// ===== TASK 1 =====

void init_data_for_task1(double*& vec, double*& matrix, double*& result, size_t size)
{
    vec = new double[size];
    matrix = new double[size * size];
    result = new double[size];

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++)
    {
        vec[i] = i;
        result[i] = 0.0;
        for (size_t j = 0; j < size; j++)
            matrix[i * size + j] = i + j;
    }
}

void run_task1(const int n, const int num_runs = 1, const int num_threads = 1)
{
    omp_set_num_threads(num_threads);
    double total_time = 0.0;
    size_t size = (size_t)n;
    double *vec = nullptr, *matrix = nullptr, *result = nullptr;

    for (int run = 0; run < num_runs + 1; ++run)
    {
        init_data_for_task1(vec, matrix, result, size);
        double start_time = omp_get_wtime();
        matrix_vector_product_task1(matrix, vec, result, size);
        if (run > 0) total_time += (omp_get_wtime() - start_time);
        free_data(vec, matrix, result);        
    }

    std::cout << total_time / num_runs << '\n';
}

// ===== TASK 2 =====

double func(double x) { return exp(-x * x); }

void run_task2(double (*func)(double), const int n, const int num_runs = 1, const int num_threads = 1)
{
    omp_set_num_threads(num_threads);
    double total_time = 0.0;

    for (int run = 0; run < num_runs + 1; ++run)
    {
        double start_time = omp_get_wtime();
        integrate(func, -4, 4, (size_t)n);
        if (run > 0) total_time += (omp_get_wtime() - start_time);        
    }

    std::cout << total_time / num_runs << '\n';
}

// ===== TASK 3 =====

void init_data_for_task3(double*& A, double*& b, double*& x, size_t n)
{
    A = new double[n * n];
    b = new double[n];
    x = new double[n];

    double c = 40.0 / n; 
    #pragma omp parallel for OMP_SCHEDULE
    for (size_t i = 0; i < n; i++)
    {
        b[i] = i + 1.0; 
        x[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            A[i * n + j] = (i == j) ? 1.0 + c : c;
    }
}

void run_task3(const int n, const int num_runs = 1, const int num_threads = 1)
{
    omp_set_num_threads(num_threads);
    double total_time_v1 = 0.0;
    double total_time_v2 = 0.0;
    double *A = nullptr, *b = nullptr, *x = nullptr;

    for (int run = 0; run < num_runs + 1; ++run)
    {
        init_data_for_task3(A, b, x, n);
        double start_time_v1 = omp_get_wtime();
        find_solution_v1(A, b, x, n);
        if (run > 0) total_time_v1 += (omp_get_wtime() - start_time_v1);
        free_data(A, b, x);

        init_data_for_task3(A, b, x, n);
        double start_time_v2 = omp_get_wtime();
        find_solution_v2(A, b, x, n);
        if (run > 0) total_time_v2 += (omp_get_wtime() - start_time_v2);
        free_data(A, b, x);
    }

    std::cout << total_time_v1 / num_runs << ' ' << total_time_v2 / num_runs << '\n';
}

// ===== BENCHMARK =====

void error_message(const char* prog_name)
{
    std::cerr << "\nInvalid arguments.\n\n"
              << "Correct usage:\n"
              << "  " << prog_name << " 1 <N> <T> <R>    # Task 1, matrix/vector size N, T threads (default 1), R runs (default 1)\n"
              << "  " << prog_name << " 2 <N> <T> <R>    # Task 2, number of integration points N, T threads (default 1), R runs (default 1)\n"
              << "  " << prog_name << " 3 <N> <T> <R>    # Task 3, matrix size N, T threads (default 1), R runs (default 1)\n\n";
    std::exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        error_message(argv[0]);
        return 1;
    }

    int task = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int num_threads = (argc >= 4) ? std::atoi(argv[3]) : 1;
    int num_runs = (argc >= 5) ? std::atoi(argv[4]) : 1;

    switch (task)
    {
        case 1:
            run_task1(N, num_runs, num_threads);
            break;
        case 2:
            run_task2(func, N, num_runs, num_threads);
            break;
        case 3:
            run_task3(N, num_runs, num_threads);
            break;
        default:
            error_message(argv[0]);
            return 1;
    }

    return 0;
}
