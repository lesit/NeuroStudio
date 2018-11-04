#if !defined(_PARALLELIZE_H)
#define _PARALLELIZE_H

/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef NP_USE_TBB
#ifndef NOMINMAX
#define NOMINMAX // tbb includes windows.h in tbb/machine/windows_api.h
#endif
#include <tbb/tbb.h>
#include <tbb/task_group.h>
#endif

#ifndef NP_USE_OMP
#include <thread>
#include <future>
#endif

namespace np
{
#ifdef USE_TBB

	static tbb::task_scheduler_init tbbScheduler(tbb::task_scheduler_init::automatic);//tbb::task_scheduler_init::deferred);

	typedef tbb::blocked_range<int> blocked_range;

	template<typename Func>
	void parallel_for(int begin, int end, const Func& f, int grainsize) {
		tbb::parallel_for(blocked_range(begin, end, end - begin > grainsize ? grainsize : 1), f);
	}
	template<typename Func>
	void xparallel_for(int begin, int end, const Func& f) {
		f(blocked_range(begin, end, 100));
	}

#else

	struct blocked_range {
		typedef int const_iterator;

		blocked_range(int begin, int end) : begin_(begin), end_(end) {}
		blocked_range(size_t begin, size_t end) : begin_(static_cast<int>(begin)), end_(static_cast<int>(end)) {}

		const_iterator begin() const { return begin_; }
		const_iterator end() const { return end_; }
	private:
		int begin_;
		int end_;
	};

	template<typename Func>
	void xparallel_for(size_t begin, size_t end, const Func& f) {
		blocked_range r(begin, end);
		f(r);
	}

#if defined(USE_OMP)

	template<typename Func>
	void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
#pragma omp parallel for
		for (int i = begin; i<end; ++i)
			f(blocked_range(i, i + 1));
	}

#elif defined(SINGLE_THREAD)

	template<typename Func>
	void parallel_for(int begin, int end, const Func& f, int /*grainsize*/) {
		xparallel_for(static_cast<size_t>(begin), static_cast<size_t>(end), f);
	}

#else

	template<typename Func>
	void parallel_for(int start, int end, const Func &f, int /*grainsize*/) {
		int nthreads = std::thread::hardware_concurrency();
		int blockSize = (end - start) / nthreads;
		if (blockSize*nthreads < end - start)
			blockSize++;

		std::vector<std::future<void>> futures;

		int blockStart = start;
		int blockEnd = blockStart + blockSize;
		if (blockEnd > end) blockEnd = end;

		for (int i = 0; i < nthreads; i++) {
			futures.push_back(std::move(std::async(std::launch::async, [blockStart, blockEnd, &f] {
				f(blocked_range(blockStart, blockEnd));
			})));

			blockStart += blockSize;
			blockEnd = blockStart + blockSize;
			if (blockStart >= end) break;
			if (blockEnd > end) blockEnd = end;
		}

		for (auto &future : futures)
			future.wait();
	}

#endif

#endif // USE_TBB

	class ParallelizeConfig
	{
	public:
		static bool IsParallelize()
		{
			return true;
		}
	};
	template<typename T, typename Func>
	inline void for_blockrange(std::true_type, int begin, T end, Func f, int grainsize = 100)
	{
		// tbb 에선 integer만 처리하기 때문에.. 근데 극복을 하긴 해야하는데...ㅠ
		if (ParallelizeConfig::IsParallelize() && value_representation<int>(end))
			parallel_for(begin, static_cast<int>(end), f, grainsize);
		else
			xparallel_for(begin, static_cast<int>(end), f);
	}

	template<typename T, typename Func>
	inline void for_blockrange(std::false_type, int begin, T end, Func f, int grainsize = 100)
	{
		if (ParallelizeConfig::IsParallelize())
			parallel_for(begin, static_cast<int>(end), f, grainsize);
		else
			xparallel_for(begin, end, f);
	}

	template<typename T, typename Func>
	inline void for_blockrange(int begin, T end, Func f, int grainsize = 100) 
	{
		static_assert(std::is_integral<T>::value, "end must be integral type");
		for_blockrange(typename std::is_unsigned<T>::type(), begin, end, f, grainsize);
	}

	template <typename T, typename Func>
	void for_i(T size, Func f, int grainsize = 100)
	{
		for_blockrange(0, size, [&](const blocked_range& r) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
			for (int i = r.begin(); i < r.end(); i++)
				f(i);
		}, grainsize);
	}
}

#endif
