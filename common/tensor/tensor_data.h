#if !defined(_TENSOR_DATA_H)
#define _TENSOR_DATA_H

#include <stdlib.h>
#include <memory.h>

#include "../data_vector.h"

namespace np
{
	namespace tensor
	{
		enum class _ts_batch_time_order{ NxTxD, TxNxD };

		template <neuro_u32 type_size>
		static bool Transpose(neuro_u32 columns, neuro_u32 rows, neuro_u32 value_size, const core::MemoryManager& target_mm, void* target, const core::MemoryManager& source_mm, const void* source)
		{
			const neuro_u32 value_bytes = value_size * type_size;

			for (neuro_u32 col = 0; col < columns; col++)
			{
				for (neuro_u32 i = 0; i < rows; i++)
				{
					if (!target_mm.Memcpy(target, (neuro_u8*)source + (i * columns + col) * value_bytes, value_bytes, source_mm))
						return false;
					target = (neuro_u8*)target + value_bytes;
				}
			}
			return true;
		}

		template <typename Type, neuro_u32 type_size>
		struct _TYPED_TENSOR_DATA
		{
		public:
			_TYPED_TENSOR_DATA(const core::MemoryManager& _mm, bool isAutoFree = false)
				: data(_mm, isAutoFree)
			{
				batch_time_order = _ts_batch_time_order::NxTxD;

				time_length = 1;

				batch_size = value_size = time_value_size = time_value_bytes = 0;
			}

			_TYPED_TENSOR_DATA(core::math_device_type mm_type = core::math_device_type::cpu, bool isAutoFree = false)
				: _TYPED_TENSOR_DATA(core::MemoryManager::GetManager(mm_type), isAutoFree)
			{

			}

			_TYPED_TENSOR_DATA& operator = (const _TYPED_TENSOR_DATA& src)
			{
				data = src.data;

				batch_time_order = src.batch_time_order;
				batch_size = src.batch_size;
				time_length = src.time_length;
				value_size = src.value_size;
				time_value_size = time_length * value_size;
				time_value_bytes = time_value_size * type_size;
				return *this;
			}

			void SetAutoFree(bool isAuto)
			{
				data.SetAutoFree(isAuto);
			}

			Type* GetBuffer() const
			{
				return data.buffer;
			}

			Type* Alloc(neuro_u32 _batch_size, neuro_u32 _value_size)
			{
				return Alloc(_batch_size, 1, _value_size);
			}

			Type* Alloc(neuro_u32 _batch_size, neuro_u32 _time_length, neuro_u32 _value_size)
			{
				if (data.Alloc(_batch_size*_time_length*_value_size))
				{
					batch_size = _batch_size;
					time_length = _time_length;
					value_size = _value_size;

					time_value_size = time_length * value_size;
					time_value_bytes = time_value_size * type_size;
				}
				else
				{
					batch_size = value_size = time_value_size = time_value_bytes = 0;
					time_length = 1;
				}

				return data.buffer;
			}

			Type* AllocLike(const _TYPED_TENSOR_DATA& src)
			{
				return Alloc(src.batch_size, src.time_length, src.value_size);
			}

			Type* Calloc(neuro_u32 _batch_size, neuro_u32 _value_size)
			{
				if (!Alloc(_batch_size, _value_size))
					return NULL;

				if (!SetZero())
				{
					Dealloc();
					return NULL;
				}
				return data.buffer;
			}

			Type* Calloc(neuro_u32 _batch_size, neuro_u32 _time_length, neuro_u32 _value_size)
			{
				if (!Alloc(_batch_size, _time_length, _value_size))
					return NULL;

				if (!SetZero())
				{
					Dealloc();
					return NULL;
				}
				return data.buffer;
			}

			void Dealloc()
			{
				data.Dealloc();
				batch_size = value_size = time_value_size = time_value_bytes = 0;
			}

			bool SetZero() const
			{
				if (data.buffer == NULL)
					return false;

				return data.mm.SetZero(data.buffer, type_size * GetSize());
			}

			bool CopyFrom(const _TYPED_TENSOR_DATA& src) const
			{
				if (data.buffer == NULL)
					return false;

				if (src.data.buffer == data.buffer)
					return true;

				if (batch_size != src.batch_size || time_value_size != src.time_value_size)
					return false;

				if (src.batch_time_order != batch_time_order)
				{
					if (batch_time_order == _ts_batch_time_order::TxNxD)	// T x N x D -> N x T x D 로 바꿀 때
						return Transpose<type_size>(batch_size, time_length, value_size, data.mm, data.buffer, src.data.mm, src.data.buffer);
					else
						return Transpose<type_size>(time_length, batch_size, value_size, data.mm, data.buffer, src.data.mm, src.data.buffer);
				}
				return data.mm.Memcpy(data.buffer, src.data.buffer, GetSize() * type_size, src.data.mm);
			}

			bool CopyFrom(const Type* src, neuro_u32 n) const
			{
				if (data.buffer == NULL)
					return false;

				if (src == data.buffer)
					return true;

				if (n != GetSize())
					return false;

				return data.mm.Memcpy(data.buffer, src, n * type_size, data.mm);
			}

			bool AllocCopyFrom(const _TYPED_TENSOR_DATA& src)
			{
				if (AllocLike(src) == NULL)
					return false;

				return CopyFrom(src);
			}

			inline neuro_u32 GetSize() const {
				return batch_size*time_length*value_size;
			}

			inline neuro_u32 GetBatchSize() const
			{
				return batch_size;
			}

			inline Type* GetBatchData(neuro_u32 offset) const
			{
				if (!CheckBatchOrderBuffer())
					return NULL;
				return (Type*)((neuro_u8*)data.buffer + offset * time_value_bytes);
			}

			inline _TYPED_DATA_VECTOR<Type, type_size> GetSample(neuro_u32 offset) const
			{
				return _TYPED_DATA_VECTOR<Type, type_size>(data.mm, GetBatchData(offset), time_value_size);
			}

			// 특수한 경우로서, batch와 time을 batch로 처리해버리는 경우이다. conv/pooling/dropout(외 normalization 등)만 해당
			inline neuro_u32 GetBatchTimeSize() const
			{
				return batch_size*time_length;
			}

			inline Type* GetBatchTimeData(neuro_u32 offset) const
			{
				if (!CheckBatchOrderBuffer())
					return NULL;
				return (Type*)((neuro_u8*)data.buffer + offset * value_size * type_size);
			}

			// fc 등 보통 time_length와 value_size가 조합되어 사용된다.
			inline neuro_u32 GetTimeValueSize() const
			{
				return time_value_size;
			}

			inline bool CheckBatchOrderBuffer() const
			{
				if (batch_time_order != _ts_batch_time_order::NxTxD)
					return false;

				if (data.buffer == NULL)
					return false;

				return true;
			}

			_ts_batch_time_order batch_time_order;

			neuro_u32 batch_size;
			neuro_u32 time_length;
			neuro_u32 value_size;
			neuro_u32 time_value_size;
			neuro_u32 time_value_bytes;
			_TYPED_DATA_VECTOR<Type, type_size> data;
		};

		typedef struct _TYPED_TENSOR_DATA<neuron_value, sizeof(neuron_value)> _NEURO_TENSOR_DATA;
	}
}

#endif
