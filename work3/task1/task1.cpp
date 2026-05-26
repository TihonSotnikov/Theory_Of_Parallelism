#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

#ifdef __linux__
#	include <sched.h>
#	include <pthread.h>
#endif

// ============================================================================
// =========================== CONSTANTS AND MACRO ============================
// ============================================================================

#if defined(__clang__) || defined(__GNUC__)
#	define RESTRICT_PTR __restrict__
#elif defined(_MSC_VER)
#	define RESTRICT_PTR __restrict
#else
#	define RESTRICT_PTR
#endif

#ifdef __cpp_lib_hardware_interference_size
constexpr size_t ALIGNMENT = std::hardware_destructive_interference_size;
#else
constexpr size_t ALIGNMENT = 64;
#endif

constexpr size_t ELEMENTS_IN_ONE_CACHE_LINE = ALIGNMENT / sizeof(double);

// ============================================================================
// ============================= BASICS PRIMITIVES ============================
// ============================================================================

class ThreadBarrier
{
private:
	const size_t            threads_num_;
	size_t                  counter_;
	size_t                  generation_{0};
	std::mutex              mtx_;
	std::condition_variable cv_;

public:
	explicit ThreadBarrier(const size_t threads_num) noexcept :
	    threads_num_(threads_num), counter_(threads_num)
	{ assert(threads_num > 0 && "Threads number must be > 0"); }

	void arrive_and_wait()
	{
		std::unique_lock<std::mutex> lock(mtx_);

		size_t const local_generation = generation_;
		counter_--;

		if (counter_ == 0)
		{
			counter_ = threads_num_;
			generation_++;
			cv_.notify_all();
		}
		else
		{
			cv_.wait(lock, [this, local_generation]() -> bool {
				return local_generation != generation_;
			});
		}
	}

	ThreadBarrier(const ThreadBarrier&) = delete;
	ThreadBarrier(ThreadBarrier&&)      = delete;

	auto operator=(const ThreadBarrier&) -> ThreadBarrier& = delete;
	auto operator=(ThreadBarrier&&) -> ThreadBarrier&      = delete;

	~ThreadBarrier() = default;
};

#ifdef USE_STD_BARRIER
#	include <barrier>
using SyncBarrier = std::barrier<>;
#else
using SyncBarrier = ThreadBarrier;
#endif

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
[[nodiscard]] auto make_aligned_array(size_t count) -> std::unique_ptr<double[], void (*)(void*)>
{
	size_t const size_bytes   = count * sizeof(double);
	size_t const aligned_size = (size_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
	// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
	void* ptr = std::aligned_alloc(ALIGNMENT, aligned_size);
	if (ptr == nullptr) throw std::bad_alloc();
	return {static_cast<double*>(ptr), std::free};
}

// ============================================================================
// =========================== NUMA & CPU PINNING =============================
// ============================================================================

inline void pin_thread_to_cpu([[maybe_unused]] const int cpu_id)
{
#if defined(__linux__) && defined(ENABLE_THREADS_PINNING_TO_CPU)
	assert(cpu_id >= 0 && cpu_id <= 79 && "CPU id must be <= 79");

	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpu_id, &mask);

	const int result = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);

	if (result)
		throw std::runtime_error("Cannot pin thread to CPU. Error: " + std::to_string(result));
#endif
}

[[nodiscard]] constexpr auto get_thread_cpu(const size_t thread_idx) noexcept -> int
{
	/*
	NUMA node0 CPU(s):         0-19,40-59
	NUMA node1 CPU(s):         20-39,60-79
	*/

	assert(thread_idx <= 79 && "Thread index must be <= 79");
	const int thread_idx_int = static_cast<int>(thread_idx);

	const int local_cpu_index = thread_idx_int / 2;
	if (thread_idx_int % 2 == 0) return (thread_idx >= 40) ? local_cpu_index + 20 : local_cpu_index;
	return (thread_idx > 40) ? local_cpu_index + 40 : local_cpu_index + 20;
}

// ============================================================================
// =============================== CALCULATIONS ===============================
// ============================================================================

[[nodiscard]] auto calculate_thread_iterations_range(const size_t curr_thread_idx,
                                                     const size_t total_threads,
                                                     const size_t total_iterations) noexcept
    -> std::pair<size_t, size_t>
{
	assert(total_threads > 0 && "Total threads must be > 0");
	assert(curr_thread_idx < total_threads && "Thread index out of range");

	size_t start_idx = 0;
	size_t end_idx   = 0;

#ifdef ENABLE_AVOIDING_FALSE_SHARING
	const size_t base_chunk = total_iterations / total_threads;
	const size_t aligned_chunk
	    = (base_chunk / ELEMENTS_IN_ONE_CACHE_LINE) * ELEMENTS_IN_ONE_CACHE_LINE;

	if (aligned_chunk == 0)
	{
		const size_t remainder = total_iterations % total_threads;
		start_idx = curr_thread_idx * base_chunk + std::min(curr_thread_idx, remainder);
		end_idx   = start_idx + base_chunk + (curr_thread_idx < remainder ? 1 : 0);
	}
	else
	{
		const size_t total_aligned = aligned_chunk * total_threads;
		const size_t remainder     = total_iterations - total_aligned;
		const size_t extra_lines   = remainder / ELEMENTS_IN_ONE_CACHE_LINE;
		const size_t leftover      = remainder % ELEMENTS_IN_ONE_CACHE_LINE;

		start_idx = curr_thread_idx * aligned_chunk
		            + std::min(curr_thread_idx, extra_lines) * ELEMENTS_IN_ONE_CACHE_LINE;
		end_idx   = (curr_thread_idx + 1) * aligned_chunk
		            + std::min(curr_thread_idx + 1, extra_lines) * ELEMENTS_IN_ONE_CACHE_LINE;

		if (curr_thread_idx == total_threads - 1 && leftover > 0) end_idx += leftover;
	}
#else
	const size_t base_chunk = total_iterations / total_threads;
	const size_t remainder  = total_iterations % total_threads;
	start_idx               = (curr_thread_idx * base_chunk) + std::min(curr_thread_idx, remainder);
	end_idx                 = start_idx + base_chunk + (curr_thread_idx < remainder ? 1 : 0);
#endif

	return {start_idx, end_idx};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline void init_one_chunk(const size_t start_idx, const size_t end_idx, const size_t n,
                           double* RESTRICT_PTR mat, double* RESTRICT_PTR vec,
                           double* RESTRICT_PTR res) noexcept
{
	assert(start_idx <= end_idx && end_idx <= n && "Bounds must be in range [0, n)");

	for (size_t i = start_idx; i < end_idx; ++i)
	{
		vec[i]                  = static_cast<double>(i);
		res[i]                  = 0.0;
		const size_t row_offset = i * n;
		for (size_t j = 0; j < n; ++j)
			mat[row_offset + j] = static_cast<double>(i) * static_cast<double>(j);
	}
}

inline void process_one_chunk(const size_t start_idx, const size_t end_idx, const size_t n,
                              const double* RESTRICT_PTR mat, const double* RESTRICT_PTR vec,
                              double* RESTRICT_PTR res) noexcept
{
	assert(start_idx <= end_idx && end_idx <= n && "Bounds must be in range [0, n)");

	for (size_t i = start_idx; i < end_idx; ++i)
	{
		double       sum        = 0.0;
		const size_t row_offset = i * n;
		for (size_t j = 0; j < n; ++j)
			sum += mat[row_offset + j] * vec[j];
		res[i] = sum;
	}
}

// ============================================================================
// ============================= THREADS CONTROL ==============================
// ============================================================================

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline void worker_routine(const size_t runs_including_warmup, const size_t start_idx,
                           const size_t end_idx, const size_t n, double* mat, double* vec,
                           double* res, const size_t threads_num, SyncBarrier& start_barrier,
                           SyncBarrier& end_barrier)
{
	assert(start_idx <= end_idx && end_idx <= n && "Bounds must be in range [0, n)");
	assert(threads_num > 0 && "Threads number must be > 0");

	init_one_chunk(start_idx, end_idx, n, mat, vec, res);

	for (size_t run = 0; run < runs_including_warmup; ++run)
	{
		start_barrier.arrive_and_wait();
		process_one_chunk(start_idx, end_idx, n, mat, vec, res);
		end_barrier.arrive_and_wait();
	}
}

// ============================================================================
// ============================ BENCHMARK AND MAIN ============================
// ============================================================================

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto benchmark(const size_t n, const size_t threads_num = 1, const size_t runs_num = 1)
    -> std::chrono::nanoseconds
{
#ifdef BENCHMARK_LOGGING
	std::cout << "\n[ BENCHMARK CONFIGURATION ]\n"
	          << "  Size (N) : " << n << '\n'
	          << "  Threads  : " << threads_num << '\n'
	          << "  Runs     : " << runs_num << "\n\n"
	          << "[ THREAD DISTRIBUTION ]\n";
#endif

	if (n == 0 || runs_num == 0) return std::chrono::nanoseconds::zero();
	if (threads_num == 0) throw std::invalid_argument("Threads number must be > 0");

	SyncBarrier start_barrier(threads_num + 1);
	SyncBarrier end_barrier(threads_num + 1);

	std::vector<std::jthread> threads;
	threads.reserve(threads_num);

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
	std::unique_ptr<double[], void (*)(void*)> const mat = make_aligned_array(n * n);
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
	std::unique_ptr<double[], void (*)(void*)> const vec = make_aligned_array(n);
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
	std::unique_ptr<double[], void (*)(void*)> const res = make_aligned_array(n);

	const size_t runs_including_warmup = runs_num + 1;

	for (size_t thread = 0; thread < threads_num; ++thread)
	{
		auto [start_idx, end_idx] = calculate_thread_iterations_range(thread, threads_num, n);

#ifdef BENCHMARK_LOGGING
		std::cout << "  Thread " << thread << " : [" << start_idx << " .. " << end_idx << ")\n";
#endif

		threads.emplace_back([=, &start_barrier, &end_barrier, mat_ptr = mat.get(),
		                      vec_ptr = vec.get(), res_ptr = res.get()]() -> void {
			pin_thread_to_cpu(get_thread_cpu(thread));
			try
			{
				worker_routine(runs_including_warmup, start_idx, end_idx, n, mat_ptr, vec_ptr,
				               res_ptr, threads_num, start_barrier, end_barrier);
			}
			catch (const std::exception& e)
			{
				std::cout << e.what() << '\n';
				std::terminate();
			}
		});
	}

	std::chrono::nanoseconds total_time = std::chrono::nanoseconds::zero();

#ifdef BENCHMARK_LOGGING
	std::cout << "\n[ EXECUTION ]\n"
	          << "  Run 0 (Warmup) : ...\n";
#endif

	for (size_t run = 0; run < runs_including_warmup; ++run)
	{
		start_barrier.arrive_and_wait();
		auto start_time = std::chrono::steady_clock::now();
		end_barrier.arrive_and_wait();

		if (run == 0) continue;

		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
		    std::chrono::steady_clock::now() - start_time);
		total_time += duration;

#ifdef BENCHMARK_LOGGING
		std::cout << "  Run " << run << "          : "
		          << std::chrono::duration<double>(duration).count() << " s\n";
#endif
	}

#ifdef BENCHMARK_LOGGING
	std::cout << "\n[ RESULTS ]\n"
	          << "  Total Time : " << std::chrono::duration<double>(total_time).count() << " s\n"
	          << "  Avg Time   : " << std::chrono::duration<double>(total_time).count() / runs_num
	          << " s\n\n";
#endif

	return total_time;
}

inline void error_message(const char* prog_name) noexcept
{
	std::cerr << "Usage:\n"
	             "   "
	          << prog_name << " <N> <threads num> <runs num>\n";
}

auto main(int argc, char* argv[]) -> int
{
	if (argc < 2)
	{
		error_message(argv[0]);
		return 1;
	}

	try
	{
		// NOLINTNEXTLINE(bugprone-unchecked-string-to-number-conversion,readability-identifier-length)
		size_t const n = atoll(argv[1]);
		// NOLINTNEXTLINE(bugprone-unchecked-string-to-number-conversion)
		size_t const threads_num = (argc >= 3) ? atoll(argv[2]) : 1;
		// NOLINTNEXTLINE(bugprone-unchecked-string-to-number-conversion)
		size_t const runs_num = (argc >= 4) ? atoll(argv[3]) : 1;

		std::cout << std::chrono::duration<double>(benchmark(n, threads_num, runs_num)).count()
		          << '\n';
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}
