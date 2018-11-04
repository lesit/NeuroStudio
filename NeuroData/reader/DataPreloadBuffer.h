#pragma once

#include "np_types.h"
#include "data_vector.h"
#include "tensor/tensor_shape.h"
namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class DataPreloadBuffer
			{
			public:
				DataPreloadBuffer();
				~DataPreloadBuffer();

				void Setup(neuro_u64 start_pos, neuro_u32 value_bytes) {
					m_start_pos = start_pos;
					m_value_bytes = value_bytes;
				}

				bool Resize(neuro_u64 count);

				bool Set(neuro_u64 pos, void* buf);
				void* Get(neuro_u64 pos);

				bool Read(neuro_u64 pos, void* buf);

				bool IsInit() const { return m_value_bytes > 0; }
				neuro_size_t GetTotalCount() const { return m_total_data; }

				void Clear();

			private:
				void* m_buffer;

				neuro_size_t m_total_data;
				neuro_u32 m_value_bytes;

				neuro_u64 m_start_pos;
			};
		}
	}
}
