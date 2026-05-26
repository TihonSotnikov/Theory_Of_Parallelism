#include "client.hpp"
#include "server.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ----- CONFIGURATION -----

constexpr double EPSILON     = 1e-5;
constexpr size_t NUM_TASKS   = 5000;
constexpr size_t NUM_THREADS = 8;

constexpr std::string SIN_FILE_NAME  = "sin_res.txt";
constexpr std::string POW_FILE_NAME  = "pow_res.txt";
constexpr std::string SQRT_FILE_NAME = "sqrt_res.txt";

using data_type = double;

// ----- CONFIGURATION -----

namespace
{

template<typename T>
bool check_results(const std::string& filename, const TaskType type, const double epsilon)
{
	std::ifstream in(filename);
	if (!in.is_open()) throw std::runtime_error("Cannot open file " + filename);

	T      arg1;
	T      arg2;
	T      res;
	size_t line = 1;

	if (type == TaskType::pow)
	{
		while (in >> arg1 >> arg2 >> res)
		{
			if (std::abs(std::pow(arg1, arg2) - res) > epsilon)
			{
				std::cerr << "Error at line " << line << '\n';
				return false;
			}
			++line;
		}
	}
	else
	{
		T (*func)(T) = (type == TaskType::sin) ? std::sin<data_type> : std::sqrt<data_type>;
		while (in >> arg1 >> res)
		{
			if (std::abs(func(arg1) - res) > epsilon)
			{
				std::cerr << "Error at line " << line << '\n';
				return false;
			}
			++line;
		}
	}
	return true;
}

} // namespace

int main()
{
	try
	{
		std::cout << "Starting server...\n";
		Server<data_type> server(NUM_THREADS);

		std::cout << "Starting clients...\n";
		{
			std::vector<std::jthread> clients;
			clients.reserve(3);
			clients.emplace_back(
			    [&server] { run_client(server, NUM_TASKS, TaskType::sin, SIN_FILE_NAME); });
			clients.emplace_back(
			    [&server] { run_client(server, NUM_TASKS, TaskType::pow, POW_FILE_NAME); });
			clients.emplace_back(
			    [&server] { run_client(server, NUM_TASKS, TaskType::sqrt, SQRT_FILE_NAME); });
		}

		std::cout << "Clients finished, stopping server...\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << "Client/server error " << e.what() << '\n';
	}

	std::cout << "Checking results...\n";

	try
	{
		std::cout << (check_results<data_type>(SIN_FILE_NAME, TaskType::sin, EPSILON)
		                  ? "\t- Sin results is correct\n"
		                  : "\t- Sin results is incorrect\n");

		std::cout << (check_results<data_type>(POW_FILE_NAME, TaskType::pow, EPSILON)
		                  ? "\t- Pow results is correct\n"
		                  : "\t- Pow results is incorrect\n");

		std::cout << (check_results<data_type>(SQRT_FILE_NAME, TaskType::sqrt, EPSILON)
		                  ? "\t- Sqrt results is correct\n"
		                  : "\t- Sqrt results is incorrect\n");
	}
	catch (const std::exception& e)
	{
		std::cerr << "Result checking error " << e.what() << '\n';
		return 1;
	}

	return 0;
}
