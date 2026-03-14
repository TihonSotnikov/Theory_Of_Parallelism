#include <iostream>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <math.h>

#include "task1.cpp"
#include "task2.cpp"
#include "task3.cpp"

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
        init_data(vec, matrix, result, m);

        double start_time = omp_get_wtime();
        matrix_vector_product(vec, matrix, result, m);
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

double func(double x) { return exp(-x * x); }

void run_task2(double (*func)(double), long long n)
{
    double T_sequential = 0.0;

    std::vector<int> nums_threads = {1, 2, 4, 6, 8};

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

void error_message(const char* prog_name)
{
    std::cerr << "\nInvalid arguments.\n\n"
              << "Correct usage:\n"
              << "  " << prog_name << " 1 M      # Task 1, matrix/vector size M\n"
              << "  " << prog_name << " 2 N      # Task 2, number of integration points N\n\n";
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
    else
    {
        error_message(argv[0]);
        return 1;  
    }

    return 0;
}