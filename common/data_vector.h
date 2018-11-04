#if !defined(_VECTOR_DATA_H)
#define _VECTOR_DATA_H

#include <stdlib.h>
#include <memory.h>

#include "np_types.h"
#include "util/np_util.h"

#include "core/MemoryManager.h"

namespace np
{
	template <typename T, neuro_u32 type_size>
	struct _TYPED_DATA_VECTOR
	{
		_TYPED_DATA_VECTOR(const core::MemoryManager& _mm, bool isAutoFree=false)
		: mm(_mm), isAutoFree(isAutoFree)
		{
			buffer = NULL;
			count = 0;
			buffer_size = 0;
		}

		_TYPED_DATA_VECTOR(core::math_device_type mm_type = core::math_device_type::cpu, bool isAutoFree = false)
			: _TYPED_DATA_VECTOR(core::MemoryManager::GetManager(mm_type), isAutoFree)
		{}

		_TYPED_DATA_VECTOR(const core::MemoryManager& _mm, T* _buffer, neuro_u32 _count)
			: mm(_mm), isAutoFree(false)
		{
			buffer = _buffer;
			count = _count;
			buffer_size = count;
		}

		_TYPED_DATA_VECTOR(const _TYPED_DATA_VECTOR& src)
			: _TYPED_DATA_VECTOR(src.mm, src.buffer, src.count)
		{
		}

		~_TYPED_DATA_VECTOR()
		{
			if (isAutoFree)
				Dealloc();
		}

		_TYPED_DATA_VECTOR& operator = (const _TYPED_DATA_VECTOR& src)
		{
			memcpy(this, &src, sizeof(_TYPED_DATA_VECTOR));
			isAutoFree = false;
			return *this;
		}

		void SetAutoFree(bool isAuto)
		{
			isAutoFree = isAuto;
		}

		T* GetBuffer() const
		{
			return buffer;
		}

		T* Alloc(neuro_u32 _count)
		{
			if (_count < count)
				return buffer;

			if (_count>buffer_size)
			{
				buffer = (T*)mm.Alloc(buffer, type_size*count, type_size*_count);
				if (buffer == NULL)
					_count = 0;

				buffer_size = _count;
			}

			count = _count;

			return buffer;
		}

		T* Calloc(neuro_u32 _count)
		{
			if (!Alloc(_count))
				return NULL;
			
			mm.SetZero(buffer, type_size * count);
			return buffer;
		}

		void Dealloc()
		{
			if (buffer)
				mm.Dealloc(buffer);
			buffer = NULL;
			count = 0;
			buffer_size = 0;
		}

		T* AddValue(const _TYPED_DATA_VECTOR& src)
		{
			if (src.count == 0)
				return buffer;

			if(Alloc(count + src.count)==NULL)
				return NULL;

			memcpy((neuro_u8*)buffer + type_size * (count - src.count), src.buffer, type_size*src.count);
			return buffer;
		}

		T* SetZero() const
		{
			if (buffer == NULL)
				return NULL;

			mm.SetZero(buffer, type_size * count);
			return buffer;
		}

		bool CopyFrom(const _TYPED_DATA_VECTOR& src) const
		{
			if (buffer == NULL)
				return false;

			if (src.buffer == buffer)
				return true;

			if (src.count != count)
				return false;

			return mm.Memcpy(buffer, src.buffer, src.count * type_size, src.mm);
		}

		const core::MemoryManager& mm;

		T* buffer;
		neuro_u32 count;
		neuro_u32 buffer_size;

	private:
		bool isAutoFree;
	};

	typedef struct _TYPED_DATA_VECTOR<neuron_value, sizeof(neuron_value)> _VALUE_VECTOR;
	typedef std::vector<_VALUE_VECTOR> _multi_value_vector;
}

#endif
