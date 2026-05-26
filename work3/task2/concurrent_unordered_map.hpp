#pragma once

#include <condition_variable>
#include <mutex>
#include <unordered_map>

template<typename Key, typename Value>
class ConcurrentUnorderedMap
{
public:
	ConcurrentUnorderedMap()  = default;
	~ConcurrentUnorderedMap() = default;

	ConcurrentUnorderedMap(const ConcurrentUnorderedMap&) = delete;
	ConcurrentUnorderedMap(ConcurrentUnorderedMap&&)      = delete;

	ConcurrentUnorderedMap& operator=(const ConcurrentUnorderedMap&) = delete;
	ConcurrentUnorderedMap& operator=(ConcurrentUnorderedMap&&)      = delete;

	void insert(Key key, Value value)
	{
		{
			const std::scoped_lock<std::mutex> lock(mtx_);
			map_.emplace(key, value);
		}
		cv_.notify_all();
	}

	Value wait_and_pop(Key key)
	{
		std::unique_lock<std::mutex> lock(mtx_);
		cv_.wait(lock, [this, &key] { return map_.find(key) != map_.end(); });
		auto node = map_.extract(key);
		return std::move(node.mapped());
	}

private:
	std::unordered_map<Key, Value> map_;
	std::mutex                     mtx_;
	std::condition_variable        cv_;
};
