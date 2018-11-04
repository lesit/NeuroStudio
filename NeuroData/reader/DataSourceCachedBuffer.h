#pragma once

#include "common.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class DataSourceCachedBuffer
			{
			public:
				DataSourceCachedBuffer(neuro_u32 column_count)
					: m_max_buffer_size(1024 * 1024 * 100)	// 최대 100mb로 한다.
					, m_column_count(column_count)
				{
					m_buffer_size = 1024;
					m_data = (neuron_value*)malloc(m_buffer_size);
					if (!m_data)
						m_buffer_size = 0;

					m_start_pos = 0;
					m_data_count = 0;
				}

				virtual ~DataSourceCachedBuffer()
				{
					if (m_data)
						free(m_data);
				}

				bool ValidPosition(neuro_u64 pos) const
				{
					return pos >= m_start_pos && pos < m_start_pos + m_data_count;
				}

				neuro_u64 GetStartPos() const { return m_start_pos; }

				neuron_value* GetData(neuro_u64 pos)
				{
					if (!ValidPosition(pos))
						return NULL;

					return m_data + (pos - m_start_pos)*m_column_count;
				}

#define _TotalBufferWritableDatacount (m_buffer_size/(m_column_count*sizeof(neuron_value)))
				neuron_value* AllocBuffer(neuro_u64 data_count, neuro_u64& available_write_data_count)
				{
					if (m_column_count == 0)
						return NULL;

					const neuro_u64 index = data_count;
					while (index >= _TotalBufferWritableDatacount)
					{
						size_t new_buffer_size = m_buffer_size + 1024;
						if (new_buffer_size > m_max_buffer_size)	// 최대치까지 할당되었다.
							return NULL;

						void* new_buffer = realloc(m_data, new_buffer_size);
						if (!new_buffer)	// 더 할당할 수 없다.
							return NULL;

						m_buffer_size = new_buffer_size;
						m_data = (neuron_value*)new_buffer;
					}

					available_write_data_count = _TotalBufferWritableDatacount - index;
					return m_data + index*m_column_count;
				}

				void SetDataInfo(neuro_u64 start_pos, neuro_u64 data_count)
				{
					m_start_pos = start_pos;
					m_data_count = data_count;
				}

			private:
				const neuro_u64 m_max_buffer_size;
				const neuro_u32 m_column_count;

				size_t m_buffer_size;

				neuron_value* m_data;
				neuro_u64 m_data_count;

				neuro_u64 m_start_pos;
			};
		}
	}
}
