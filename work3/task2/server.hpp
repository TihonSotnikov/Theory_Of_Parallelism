#pragma once

#include "concurrent_unordered_map.hpp"

#include "thread_pool.hpp"
#include <atomic>
#include <cstddef>
#include <functional>

template<typename T>
class Server
{
public:
	explicit Server(size_t num_threads) : pool_(num_threads) {}

	void start() = delete;
	void stop()  = delete;

	size_t add_task(const std::function<T()>& task)
	{
		size_t const id             = task_counter_++;
		auto         server_wrapper = [task, id, this] {
			T result = task();
			results_.insert(id, result);
		};
		pool_.enqueue(server_wrapper);
		return id;
	}

	T request_result(size_t id) { return results_.wait_and_pop(id); }

private:
	ConcurrentUnorderedMap<size_t, T> results_;
	std::atomic<size_t>               task_counter_{0};
	ThreadPool                        pool_;
};
