#pragma once

#include "concurrent_queue.hpp"

#include <cstddef>
#include <functional>
#include <future>
#include <optional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool
{
public:
	explicit ThreadPool(size_t num_threads)
	{
		workers_.reserve(num_threads);
		for (size_t i = 0; i < num_threads; ++i)
		{
			workers_.emplace_back([this](const std::stop_token& stoken) {
				while (true)
				{
					std::optional<std::function<void()>> task = tasks_.wait_and_pop(stoken);
					if (task == std::nullopt) return;
					(*task)();
				}
			});
		}
	}
	~ThreadPool()
	{
		for (auto& worker : workers_)
			worker.request_stop();
	}

	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&)      = delete;

	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&&)      = delete;

	template<class F, typename... Args>
	auto enqueue(F&& f, Args&&... args)
	    -> std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>
	{
		using return_type = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
		auto task         = std::make_shared<std::packaged_task<return_type()>>(
		    [f        = std::forward<F>(f),
		     ... args = std::forward<Args>(args...)]() mutable -> return_type {
			    return f(std::move(args)...);
		    });
		std::future<return_type> task_future = task->get_future();
		tasks_.push([task] { (*task)(); });
		return task_future;
	}

private:
	ConcurrentQueue<std::function<void()>> tasks_;
	std::vector<std::jthread>              workers_;
};
