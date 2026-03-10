#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <threads> <version(1 or 2)>" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1]);
    int threads = std::stoi(argv[2]);
    int version = std::stoi(argv[3]);

    omp_set_num_threads(threads);

    std::vector<double> A((size_t)N * N);
    std::vector<double> b(N);
    std::vector<double> x(N);
    std::vector<double> Ax(N);

    // Параллельная инициализация
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        b[i] = N + 1.0;
        x[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            A[(size_t)i * N + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    double norm_b_sq = 0.0;
    #pragma omp parallel for reduction(+:norm_b_sq)
    for (int i = 0; i < N; ++i) {
        norm_b_sq += b[i] * b[i];
    }
    double norm_b = std::sqrt(norm_b_sq);

    // Аналитически вычисленный оптимальный шаг
    double tau = 2.0 / (N + 2.0);
    double eps = 1e-5;
    
    int iters = 0;
    double start_time = omp_get_wtime();

    if (version == 1) {
        // Версия 1: #pragma omp parallel for на каждый цикл
        while (true) {
            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                double sum = 0.0;
                for (int j = 0; j < N; ++j) {
                    sum += A[(size_t)i * N + j] * x[j];
                }
                Ax[i] = sum;
            }

            double norm_num_sq = 0.0;
            #pragma omp parallel for reduction(+:norm_num_sq)
            for (int i = 0; i < N; ++i) {
                double diff = Ax[i] - b[i];
                norm_num_sq += diff * diff;
            }

            if (std::sqrt(norm_num_sq) / norm_b < eps) {
                break;
            }

            #pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                x[i] -= tau * (Ax[i] - b[i]);
            }
            iters++;
        }
    } 
    else if (version == 2) {
        // Версия 2: одна параллельная область снаружи
        bool stop = false;
        double norm_num_sq = 0.0;

        #pragma omp parallel
        {
            while (!stop) {
                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < N; ++j) {
                        sum += A[(size_t)i * N + j] * x[j];
                    }
                    Ax[i] = sum;
                }

                #pragma omp single
                norm_num_sq = 0.0;
                // Неявный барьер после single

                #pragma omp for reduction(+:norm_num_sq)
                for (int i = 0; i < N; ++i) {
                    double diff = Ax[i] - b[i];
                    norm_num_sq += diff * diff;
                }

                #pragma omp single
                {
                    if (std::sqrt(norm_num_sq) / norm_b < eps) {
                        stop = true;
                    } else {
                        iters++;
                    }
                }
                // Неявный барьер гарантирует, что все потоки увидят актуальный stop

                if (stop) break;

                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    x[i] -= tau * (Ax[i] - b[i]);
                }
            }
        }
    }

    double end_time = omp_get_wtime();
    double time_spent = end_time - start_time;

    std::cout << "Version: " << version 
              << " | N: " << N 
              << " | Threads: " << threads 
              << " | Iters: " << iters 
              << " | Time: " << time_spent << " s" << std::endl;

    return 0;
}