#include "stdafx.h"

#include "MiniBatchGenerator.h"

using namespace np;
using namespace np::engine;

MiniBatchGenerator::MiniBatchGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector)
	: m_provider(provider), m_data_vector(data_vector)
{
	m_batch_count = 0;

	m_position = 0;
	m_total_data_count = provider.GetDataCount();

	m_is_sequence_read = false;
}

MiniBatchGenerator::~MiniBatchGenerator()
{
	m_cpu_read_buffer.Dealloc();
}

bool MiniBatchGenerator::Ready(neuro_u32 batch_size)
{
	m_batch_size = m_data_vector.size() > 0 ? batch_size : 0;

	if (m_batch_size == 0)
	{
		DEBUG_OUTPUT(L"no batch");
		return false;
	}

	if (m_data_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no target data buffer");
		return false;
	}

	if (!m_provider.Preload())
	{
		DEBUG_OUTPUT(L"failed preload of input provider");
		return false;
	}

	DEBUG_OUTPUT(L"total data=%llu, batch=%u", m_total_data_count, m_batch_size);

	neuro_u32 max_time_length = 1;
	for (neuro_size_t i = 0; i < m_data_vector.size(); i++)
	{
		const _PRODUCER_LAYER_DATA_SET& producer_layer_data_set = m_data_vector[i];

		const neuro_u32 layer_data_size = producer_layer_data_set.layer_data_size;
		const neuro_u32 producer_dim_size = producer_layer_data_set.read_label ? 1 : producer_layer_data_set.producer->m_data_dim_size;

		neuro_u32 time_length = layer_data_size / producer_dim_size;
		max_time_length = max(max_time_length, time_length);
	}
	if (m_total_data_count <= max_time_length - 1)
	{
		DEBUG_OUTPUT(L"time length[%u] is over than total data count[%u]", m_total_data_count, max_time_length);
		return false;
	}
	m_total_data_count -= max_time_length - 1;

	if (m_batch_size > m_total_data_count)
	{
		DEBUG_OUTPUT(L"batch=%u -> %u because it is over total data size", m_batch_size, m_total_data_count);
		m_batch_size = m_total_data_count;
	}

	DEBUG_OUTPUT(L"total data=%llu, batch=%u", m_total_data_count, m_batch_size);

	if (m_batch_size == 0)
	{
		DEBUG_OUTPUT(L"invalid parametered. batch size is zero");
		m_batch_count = 0;
		return false;
	}

	m_batch_count = NP_Util::CalculateCountPer(m_total_data_count, m_batch_size);
	m_position = 0;

	return true;
}

neuro_u32 MiniBatchGenerator::ReadBatchData(bool is_learn_mode)
{
	Timer timer;
	neuro_u32 read = ReadBatchDataFromProvider(is_learn_mode);
	double elapse = timer.elapsed();
	if (elapse>1)
		DEBUG_OUTPUT(L"read input. pos[%llu], elapse : %f", m_position, elapse);

	if (read == 0)
	{
		DEBUG_OUTPUT(L"failed ReadBatchData for input. pos[%llu]", m_position);
		return 0;
	}

	if (m_is_sequence_read)
		++m_position;
	else
		m_position += read;

	if (m_position >= m_total_data_count)
		m_position = m_position % m_total_data_count;
	return read;
}

inline neuro_u32 MiniBatchGenerator::ReadBatchDataFromProvider(bool is_learn_mode)
{
	if (m_data_vector.size() == 0)
		return 0;

	neuro_u32 read_data = 0;
	for (neuro_size_t i = 0; i < m_data_vector.size(); i++)
	{
		read_data = 0;

		const _PRODUCER_LAYER_DATA_SET& producer_layer_data_set = m_data_vector[i];

		void* layer_buffer = producer_layer_data_set.layer_buffer;
		if (layer_buffer == NULL)
		{
			DEBUG_OUTPUT(L"no layer buffer");
			return 0;
		}

		dp::preprocessor::AbstractProducer* producer = producer_layer_data_set.producer;

		const neuro_u32 layer_data_size = producer_layer_data_set.layer_data_size;
		const neuro_u32 sample_bytes = layer_data_size * sizeof(neuro_float);

		const neuro_u32 producer_dim_size = producer_layer_data_set.read_label ? 1 : producer->m_data_dim_size;
		const neuro_u32 producer_dim_bytes = producer_dim_size * sizeof(neuro_float);

		const neuro_size_t last_distance = layer_data_size / producer_dim_size;

		void* read_buffer;

		const core::MemoryManager& target_mm = *producer_layer_data_set.layer_mm;
		if (target_mm.GetType() != core::math_device_type::cpu)
		{
			m_cpu_read_buffer.Alloc(layer_data_size * m_batch_size);
			read_buffer = m_cpu_read_buffer.buffer;
		}
		else
			read_buffer = layer_buffer;

		void* cur_buffer = read_buffer;

		neuro_size_t pos = m_position;
		for (neuro_size_t sample = 0; sample<m_batch_size; sample++, cur_buffer = (neuro_u8*)cur_buffer + sample_bytes)
		{
			neuro_size_t read_pos = GetRelativePosition(pos);
			neuro_size_t last_pos = read_pos + last_distance;
			void* buffer = cur_buffer;
			while (read_pos < last_pos)
			{
				if (producer_layer_data_set.read_label)
				{
					neuro_u32 label;
					if (!producer->ReadLabel(read_pos, label))
					{
						DEBUG_OUTPUT(L"failed read label. pos:%u", read_pos);
						return 0;
					}
#ifdef _DEBUG
					if (sample == 0)
						int a = 0;
#endif
					((neuro_u32*)buffer)[0] = label;
				}
				else if (!producer->Read(read_pos, (neuro_float*)buffer))
				{
					DEBUG_OUTPUT(L"failed read. pos:%u", read_pos);
					return 0;
				}
				buffer = (neuro_u8*)buffer + producer_dim_bytes;
				++read_pos;
			}

			++read_data;
			++pos;
			if (pos == m_total_data_count)
			{
				// begin reading from the beginning when learn mode
				if (is_learn_mode)
					pos = 0;
				else
					break;
			}
		}
		if (read_buffer == m_cpu_read_buffer.buffer)
			target_mm.Memcpy(layer_buffer, read_buffer, sample_bytes * m_batch_size, core::math_device_type::cpu);
	}

//#define _BATCH_GEN_DISPLAY_VALUES
#ifdef _BATCH_GEN_DISPLAY_VALUES
	if (is_input && m_position % 100 == 0 && sample == 0)
	{
		for (int i = 0; i < 5; i++)
			NP_Util::DebugOutputValues(m_cpu_read_buffer.buffer + i * 51, 10, 10);
		NP_Util::DebugOutput(L"\r\n");
	}
#endif

	return read_data;
}

MiniBatchSequentialGenerator::MiniBatchSequentialGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector)
	:MiniBatchGenerator(provider, data_vector)
{
}

MiniBatchSequentialGenerator::~MiniBatchSequentialGenerator()
{
}

#include "util/randoms.h"
MiniBatchShuffleGenerator::MiniBatchShuffleGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector)
	:MiniBatchSequentialGenerator(provider, data_vector)
{
}

MiniBatchShuffleGenerator::~MiniBatchShuffleGenerator()
{
}

bool MiniBatchShuffleGenerator::Ready(neuro_u32 batch_size)
{
	if (!__super::Ready(batch_size))
		return false;

	m_permutation.resize(m_total_data_count);
	for (neuro_size_t i = 0; i < m_total_data_count; i++)
		m_permutation[i] = i;

	random_generator& g = random_generator::get_instance();
	std::shuffle(m_permutation.begin(), m_permutation.end(), g());

	return true;
}
