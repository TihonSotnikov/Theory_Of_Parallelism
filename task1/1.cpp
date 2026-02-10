#include <iostream>
#include <cmath>
#include <vector>
#include <numbers>

constexpr std::size_t ARR_LEN = 10'000'000;

#ifdef USE_DOUBLE
    using a_type = double;
    constexpr const char* a_type_name = "double";
#else
    using a_type = float;
    constexpr const char* a_type_name = "float";
#endif

std::vector<a_type> fill_array()
{
    std::vector<a_type> sin_values_arr(ARR_LEN);
    constexpr a_type step = a_type(2) * std::numbers::pi_v<a_type> / a_type(ARR_LEN);
    for (std::size_t i = 0; i < ARR_LEN; ++i)
        sin_values_arr[i] = std::sin(step * a_type(i));
    return sin_values_arr;
}

double calc_sum(const std::vector<a_type>& sin_values_arr)
{
    double sum = a_type(0);
    for (auto val : sin_values_arr)
        sum += val;
    return sum;
}

int main()
{
    std::cout << "Summa of Sin values using type " << a_type_name << ":\n";
    std::cout << calc_sum(fill_array()) << std::endl;
    return 0;
}
