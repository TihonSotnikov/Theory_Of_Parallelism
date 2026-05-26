#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <stop_token>
#include <utility>

template<typename T>
class ConcurrentQueue
{
private:
	std::queue<T>               queue_;
	mutable std::mutex          mtx_;
	std::condition_variable_any cv_;
	bool                        closed_ = false;

public:
	ConcurrentQueue()  = default;
	~ConcurrentQueue() = default;

	ConcurrentQueue(const ConcurrentQueue&) = delete;
	ConcurrentQueue(ConcurrentQueue&&)      = delete;

	ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;
	ConcurrentQueue& operator=(ConcurrentQueue&&)      = delete;

	[[nodiscard]] bool empty() const
	{
		std::scoped_lock lock(mtx_);
		return queue_.empty();
	}

	template<class U>
	void push(U&& value)
	{
		{
			const std::scoped_lock lock(mtx_);
			if (closed_) return;
			queue_.push(std::forward<U>(value));
		}
		cv_.notify_one();
	}

	std::optional<T> wait_and_pop(std::stop_token stoken)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		cv_.wait(lock, std::move(stoken), [this] { return !queue_.empty() || closed_; });
		if (queue_.empty()) return std::nullopt;
		std::optional<T> value = std::move(queue_.front());
		queue_.pop();
		return value;
	}

	std::optional<T> try_pop()
	{
		std::scoped_lock lock(mtx_);
		if (queue_.empty()) return std::nullopt;
		std::optional<T> value = std::move(queue_.front());
		queue_.pop();
		return value;
	}

	void close()
	{
		{
			std::scoped_lock lock(mtx_);
			closed_ = true;
		}
		cv_.notify_all();
	}
};
