#pragma once

#include "server.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

template<typename T>
T fun_sin(T arg)
{ return std::sin(arg); }

template<typename T>
T fun_sqrt(T arg)
{ return std::sqrt(arg); }

template<typename T>
T fun_pow(T x, T y)
{ return std::pow(x, y); }

enum class TaskType : uint8_t
{
	sin,
	sqrt,
	pow
};

template<typename T>
void run_client(Server<T>& server, size_t task_count, TaskType type, const std::string& filename)
{
	std::ofstream out(filename);
	if (!out.is_open()) throw std::runtime_error("Cannot to open file " + filename);
	out.precision(16);

	std::mt19937                      rng(std::random_device{}());
	std::uniform_real_distribution<T> dist(1.0, 9.0);

	std::vector<std::tuple<size_t, T, T>> results_ids;
	results_ids.reserve(task_count);

	for (size_t i = 0; i < task_count; ++i)
	{
		T arg1 = dist(rng);
		T arg2 = dist(rng);

		std::function<T()> task;
		switch (type)
		{
		case TaskType::sin:
			task = [arg1] { return fun_sin(arg1); };
			break;
		case TaskType::pow:
			task = [arg1, arg2] { return fun_pow(arg1, arg2); };
			break;
		case TaskType::sqrt:
			task = [arg1] { return fun_sqrt(arg1); };
			break;
		}

		results_ids.emplace_back(server.add_task(task), arg1, arg2);
	}

	for (const auto& [id, arg1, arg2] : results_ids)
	{
		T res = server.request_result(id);
		if (type == TaskType::pow) out << arg1 << ' ' << arg2 << ' ' << res << '\n';
		else out << arg1 << ' ' << res << '\n';
	}
}
