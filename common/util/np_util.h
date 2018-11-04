#if !defined(_NP_UTIL_H)
#define _NP_UTIL_H

#include "../np_types.h"
#include <algorithm>
#include <unordered_map>
#include <chrono>

namespace np
{
	class NP_Util
	{
	public:
		static neuro_u64 CalculateCountPer(neuro_u64 base, neuro_u64 devide)
		{
			if (devide == 0)
				return 0;

			return (base + devide - 1) / devide;
		}

		static std::wstring TimeOutput();

		static void SetDebugLogWriteFile(const wchar_t* path);

		static void DebugOutput(const wchar_t* strFormat, ...);
		static void DebugOutputWithFunctionName(const char* func_name, const wchar_t* strFormat, ...);

		static void DebugOutputValues(const neuron_value* buffer, neuro_size_t count, int line_size);
		static void DebugOutputValues(const neuro_u32* buffer, neuro_size_t count, int line_size);

		static neuro_u64 GetAvailableMemory();

		static std::string GetSizeString(neuro_size_t size);

	protected:
		static void VDebugOutput(const std::wstring* time, const wchar_t* func_name, const wchar_t* strFormat, va_list args);
	};

	class Timer
	{
	public:
		Timer() : t1(std::chrono::high_resolution_clock::now()){};
		double elapsed(){ return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count(); }
		void restart(){ t1 = std::chrono::high_resolution_clock::now(); }
		void stop(){ t2 = std::chrono::high_resolution_clock::now(); }
		double total(){ stop(); return std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count(); }
		~Timer(){}
	private:
		std::chrono::high_resolution_clock::time_point t1, t2;
	};

	template<typename T>
	static void GetMinMax(const T* buffer, neuro_size_t size, T& min, T& max)
	{
		if (size == 0)
		{
			min = max = 0;
			return;
		}
		min = max = *buffer;
		for (neuro_size_t i = 1; i < size; i++)
		{
			if (min > buffer[i])
				min = buffer[i];
			if (max < buffer[i])
				max = buffer[i];
		}
	}

	template<typename type, neuro_u32 D>
	struct _TYPED_VECTOR_N_DEF
	{
		void SetVector(const std::vector<type>& src)
		{
			neuro_size_t i = 0, n = min(D, (type)src.size());
			for (; i < n; i++)
				dims[i] = src[i];
			for (; i < D; i++)
				dims[i] = 0;
		}

		void GetVector(std::vector<type>& ret) const
		{
			ret.clear();
			for (neuro_size_t i = 0; i < D && dims[i] != 0; i++)
				ret.push_back(dims[i]);
		}

		neuro_u32 dims[D];//0이 아닐때까지가 유효한 dimension. 즉, [5][7][2][0][0] 이런식으로 정의됨
	};

	/*
	__int64 i11=999999999999994LL;	// 완전 정확
	neuro_float d21=i11;
	__int64 i21=d21;

	__int64 i12=9999999999999995LL;		// 정확. 마지막 한자리가 5이상이면 1올림
	neuro_float d22=i12;
	__int64 i22=d22;

	__int64 i13=99999999999999994LL;	// 마지막 한자리 무조건 올림
	neuro_float d23=i13;
	__int64 i23=d23;

	__int64 i14=999999999999999994LL;	// 마지막 두자리 불일치. 즉, 마지막 자리에서 올림
	neuro_float d24=i14;
	__int64 i24=d24;

	__int32 org32=2147483647L;
	__int64 org64=9223372036854775807LL;
	*/

	typedef std::vector<neuro_float> neuro_data_vector;

	template<typename T>
	inline void value_copy(T* target, const T* source, neuro_size_t count)
	{
		memcpy(target, source, sizeof(T)*count);
	}

	template<typename T>
	T* reverse_endian(T* p) {
		std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p)+sizeof(T));
		return p;
	}

	inline bool is_little_endian() {
		int x = 1;
		return *(char*)&x != 0;
	}

	template<typename T>
	size_t max_index(const T* buf, size_t count)
	{
		size_t max = 0;
		for (size_t index = 1; index < count; index++)
			if (buf[index]>buf[max])
				max = index;
		return max;
	}

	template<typename T>
	size_t max_index(const T* begin, const T* end) 
	{
		return max_index(begin, end - begin);
	}

	template<typename T>
	size_t max_index(const T& vec) {
		auto begin = std::begin(vec);
		auto end = std::end(vec);
		return max_index(*begin, end - begin);
	}

	template<typename T, typename U>
	U rescale(T x, T src_min, T src_max, U dst_min, U dst_max) {
		return static_cast<U>(((x - src_min) * (dst_max - dst_min)) / (src_max - src_min) + dst_min);
	}

	inline void nop()
	{
		// do nothing
	}

	template<typename T, typename U>
	bool value_representation(U const &value) {
		return static_cast<U>(static_cast<T>(value)) == value;
	}

	template <typename T>
	int compare(const void* a, const void* b)
	{
		const T& first = *((T*)a);
		const T& second = *((T*)b);
		if (first<second)
			return -1;
		else if (first>second)
			return 1;
		else
			return 0;
	}

	template <typename T>
	int reverse_compare(const void* a, const void* b)
	{
		return -compare<T>(a, b);
	}

	template <typename T>
	void sort(std::vector<T>& v, bool bReverse = false)
	{
		if (bReverse)
			std::qsort(&v[0], v.size(), sizeof(T), reverse_compare<T>);
		else
			std::qsort(&v[0], v.size(), sizeof(T), compare<T>);
	}
	/*
	template<typename T>
	bool valid_number_range(neuro_u64 value)
	{
		//typeid(T).name()
		return value > std::numeric_limits<T>::max();
	}*/
}

#define htonll(x) \
	((((x)& 0xff00000000000000LL) >> 56) | \
	(((x)& 0x00ff000000000000LL) >> 40) | \
	(((x)& 0x0000ff0000000000LL) >> 24) | \
	(((x)& 0x000000ff00000000LL) >> 8) | \
	(((x)& 0x00000000ff000000LL) << 8) | \
	(((x)& 0x0000000000ff0000LL) << 24) | \
	(((x)& 0x000000000000ff00LL) << 40) | \
	(((x)& 0x00000000000000ffLL) << 56))

#define ntohll(x) \
	((((x)& 0x00000000000000FFLL) << 56) | \
	(((x)& 0x000000000000FF00LL) << 40) | \
	(((x)& 0x0000000000FF0000LL) << 24) | \
	(((x)& 0x00000000FF000000LL) << 8) | \
	(((x)& 0x000000FF00000000LL) >> 8) | \
	(((x)& 0x0000FF0000000000LL) >> 24) | \
	(((x)& 0x00FF000000000000LL) >> 40) | \
	(((x)& 0xFF00000000000000LL) >> 56))

#ifdef _WIN32
#define burn(mem,size) do { volatile char *burnm = (volatile char *)(mem); int burnc = size; RtlSecureZeroMemory (mem, size); while (burnc--) *burnm++ = 0; } while (0)
#else
#define burn(mem,size) do { volatile char *burnm = (volatile char *)(mem); int burnc = size; while (burnc--) *burnm++ = 0; } while (0)
#endif

#if !defined(DEBUG_OUTPUT)
#define DEBUG_OUTPUT(fmt, ...)	np::NP_Util::DebugOutputWithFunctionName(__FUNCTION__, fmt, __VA_ARGS__)
/*
#ifdef _DEBUG
#define DEBUG_OUTPUT(fmt, ...)	np::NP_Util::DebugOutputWithFunctionName(__FUNCTION__, fmt, __VA_ARGS__)
#else
#define DEBUG_OUTPUT(fmt, ...)
#endif
*/
#endif

#endif
