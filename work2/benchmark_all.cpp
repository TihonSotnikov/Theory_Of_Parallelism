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

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        vec[i] = i;
        result[i] = 0.0;
        for (size_t j = 0; j < size; j++)
            matrix[i * size + j] = i + j;
    }
}

void run_task1(int m)
{
    double *vec = nullptr, *matrix = nullptr, *result = nullptr;
    double T_sequential = 0.0;

    std::vector<int> nums_threads = {1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << "Threads  Time(s)  Speedup\n";
    std::cout << "-------------------------\n";
    std::cout << std::fixed << std::setprecision(5);

    for (int cur_num_threads : nums_threads)
    {
        omp_set_num_threads(cur_num_threads);

        init_data_for_task1(vec, matrix, result, m);

        double start_time = omp_get_wtime();
        matrix_vector_product_task1(vec, matrix, result, m);
        double elapsed_time = omp_get_wtime() - start_time;

        if (cur_num_threads == 1)
            T_sequential = elapsed_time;

        double speedup = T_sequential / elapsed_time;

        std::cout << cur_num_threads << "        "
                  << elapsed_time << "  "
                  << speedup << '\n';

        free_data(vec, matrix, result);
    }
}

// ===== TASK 2 =====

double func(double x) { return exp(-x * x); }

void run_task2(double (*func)(double), long long n)
{
    double T_sequential = 0.0;

    std::vector<int> nums_threads = {1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << "Threads  Time(s)  Speedup\n";
    std::cout << "-------------------------\n";
    std::cout << std::fixed << std::setprecision(5);

    for (int cur_num_threads : nums_threads)
    {
        omp_set_num_threads(cur_num_threads);

        double start_time = omp_get_wtime();
        integrate(func, -4, 4, n);
        double elapsed_time = omp_get_wtime() - start_time;

        if (cur_num_threads == 1)
            T_sequential = elapsed_time;

        double speedup = T_sequential / elapsed_time;

        std::cout << cur_num_threads << "        "
                  << elapsed_time << "  "
                  << speedup << '\n';
    }
}

// ===== TASK 3 =====

void init_data_for_task3(double*& A, double*& b, double*& x, size_t n)
{
    A = new double[n * n];
    b = new double[n];
    x = new double[n];

    double b_val = n + 1;
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        b[i] = b_val;
        x[i] = 0.0;
        for (size_t j = 0; j < n; j++)
            A[i * n + j] = (i == j) ? 2.0 : 1.0;
    }
}

void run_task3(size_t n)
{
    double *A = nullptr, *x = nullptr, *b = nullptr;
    double T_seq_v1 = 0.0, T_seq_v2 = 0.0;

    std::vector<int> nums_threads = {1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << "Threads  Time_v1(s)  Speedup_v1  Time_v2(s)  Speedup_v2\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(5);

    for (int cur_num_threads : nums_threads)
    {
        omp_set_num_threads(cur_num_threads);

        // Замер Варианта 1
        init_data_for_task3(A, b, x, n);
        double start_time1 = omp_get_wtime();
        find_solution_v1(A, b, x, n);
        double time_v1 = omp_get_wtime() - start_time1;
        
        if (cur_num_threads == 1) T_seq_v1 = time_v1;
        double speedup_v1 = T_seq_v1 / time_v1;
        free_data(A, b, x);

        // Замер Варианта 2
        init_data_for_task3(A, b, x, n);
        double start_time2 = omp_get_wtime();
        find_solution_v2(A, b, x, n);
        double time_v2 = omp_get_wtime() - start_time2;
        
        if (cur_num_threads == 1) T_seq_v2 = time_v2;
        double speedup_v2 = T_seq_v2 / time_v2;
        free_data(A, b, x);

        std::cout << std::setw(7) << cur_num_threads << "  "
                  << std::setw(10) << time_v1 << "  "
                  << std::setw(10) << speedup_v1 << "  "
                  << std::setw(10) << time_v2 << "  "
                  << std::setw(10) << speedup_v2 << '\n';
    }
}

// ===== BENCHMARK =====

void error_message(const char* prog_name)
{
    std::cerr << "\nInvalid arguments.\n\n"
              << "Correct usage:\n"
              << "  " << prog_name << " 1 M      # Task 1, matrix/vector size M\n"
              << "  " << prog_name << " 2 N      # Task 2, number of integration points N\n"
              << "  " << prog_name << " 3 N      # Task 3, matrix size N\n\n";
    std::exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        error_message(argv[0]);
        return 1;
    }
    if (*argv[1] == '1') run_task1(atoi(argv[2]));
    else if (*argv[1] == '2') run_task2(func, atoll(argv[2]));
    else if (*argv[1] == '3') run_task3(atoll(argv[2]));
    else
    {
        error_message(argv[0]);
        return 1;
    }

    return 0;
}