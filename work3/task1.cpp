#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#	define RESTRICT_PTR __restrict__
#elif defined(_MSC_VER)
#	define RESTRICT_PTR __restrict
#else
#	define RESTRICT_PTR
#endif

class ThreadBarrier
{
private:
	const size_t            threads_num;
	size_t                  counter;
	size_t                  generation;
	std::mutex              mtx;
	std::condition_variable cv;

public:
	explicit ThreadBarrier(const size_t threads_num) :
	    threads_num(threads_num), counter(threads_num), generation(0)
	{ assert(threads_num > 0 && "Threads number must be > 0"); }

	void wait()
	{
		std::unique_lock<std::mutex> lock(mtx);

		size_t local_generation = generation;
		counter--;

		if (counter == 0)
		{
			counter = threads_num;
			generation++;
			cv.notify_all();
		}
		else
		{
			cv.wait(lock, [this, local_generation] { return local_generation != generation; });
		}
	}

	ThreadBarrier(const ThreadBarrier&)            = delete;
	ThreadBarrier& operator=(const ThreadBarrier&) = delete;
};

[[nodiscard]] constexpr std::pair<size_t, size_t> calculate_thread_iterations_range(
    const size_t curr_thread_idx, const size_t total_threads, const size_t total_iterations)
{
	assert(total_threads > 0 && "Total threads must be > 0");
	assert(curr_thread_idx < total_threads && "Thread index out of range");

	const size_t base_chunk = total_iterations / total_threads;
	const size_t remainder  = total_iterations % total_threads;
	const size_t start_idx  = curr_thread_idx * base_chunk + std::min(curr_thread_idx, remainder);
	const size_t end_idx    = start_idx + base_chunk + (curr_thread_idx < remainder ? 1 : 0);

	return {start_idx, end_idx};
}

inline void init_one_chunk(const size_t start_idx, const size_t end_idx, const size_t n,
                           double* RESTRICT_PTR mat, double* RESTRICT_PTR vec,
                           double* RESTRICT_PTR res)
{
	assert(start_idx <= end_idx && end_idx <= n && "Bounds must be in range [0, n)");

	for (size_t i = start_idx; i < end_idx; ++i)
	{
		vec[i]                  = i;
		res[i]                  = 0.0;
		const size_t row_offset = i * n;
		for (size_t j = 0; j < n; ++j)
			mat[row_offset + j] = i * j;
	}
}

inline void process_one_chunk(const size_t start_idx, const size_t end_idx, const size_t n,
                              const double* RESTRICT_PTR mat, const double* RESTRICT_PTR vec,
                              double* RESTRICT_PTR res)
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

inline void worker_routine(const size_t runs_num, const size_t start_idx, const size_t end_idx,
                           const size_t n, double* mat, double* vec, double* res,
                           const size_t threads_num, ThreadBarrier& start_barrier,
                           ThreadBarrier& end_barrier)
{
	assert(start_idx <= end_idx && end_idx <= n && "Bounds must be in range [0, n)");
	assert(threads_num > 0 && "Threads number must be > 0");

	init_one_chunk(start_idx, end_idx, n, mat, vec, res);

	for (size_t run = 0; run < runs_num; ++run)
	{
		start_barrier.wait();
		process_one_chunk(start_idx, end_idx, n, mat, vec, res);
		end_barrier.wait();
	}
}

std::chrono::nanoseconds benchmark(const size_t n, const size_t threads_num = 1,
                                   const size_t runs_num = 1)
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

	ThreadBarrier start_barrier(threads_num + 1);
	ThreadBarrier end_barrier(threads_num + 1);

	std::vector<std::jthread> threads;
	threads.reserve(threads_num);

	std::unique_ptr<double[]> mat = std::make_unique_for_overwrite<double[]>(n * n);
	std::unique_ptr<double[]> vec = std::make_unique_for_overwrite<double[]>(n);
	std::unique_ptr<double[]> res = std::make_unique_for_overwrite<double[]>(n);

	const size_t actual_runs_num = runs_num + 1;

	for (size_t thread = 0; thread < threads_num; ++thread)
	{
		auto [start_idx, end_idx] = calculate_thread_iterations_range(thread, threads_num, n);

#ifdef BENCHMARK_LOGGING
		std::cout << "  Thread " << thread << " : [" << start_idx << " .. " << end_idx << ")\n";
#endif

		threads.emplace_back(
		    [=, &start_barrier, &end_barrier, mat_ptr = mat.get(), vec_ptr = vec.get(),
		     res_ptr = res.get()]
		    {
			    try
			    {
				    worker_routine(actual_runs_num, start_idx, end_idx, n, mat_ptr, vec_ptr,
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

	for (size_t run = 0; run < actual_runs_num; ++run)
	{
		start_barrier.wait();
		auto start_time = std::chrono::steady_clock::now();
		end_barrier.wait();

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

inline void error_message(const char* prog_name)
{
	std::cout << "Usage:\n"
	             "   "
	          << prog_name << " <N> <threads num> <runs num>";
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		error_message(argv[0]);
		return 1;
	}

	size_t n           = atoll(argv[1]);
	size_t threads_num = (argc >= 3) ? atoll(argv[2]) : 1;
	size_t runs_num    = (argc >= 4) ? atoll(argv[3]) : 1;

	std::cout << std::chrono::duration<double>(benchmark(n, threads_num, runs_num)).count() << '\n';

	return 0;
}
