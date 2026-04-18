#include <omp.h>

double integrate(double (*func)(double), double a, double b, size_t n)
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel
    {
        double sumloc = 0.0;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++)
            sumloc += func(a + h * (i + 0.5));

        #pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}
