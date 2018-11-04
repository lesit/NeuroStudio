#pragma once

#include "common.h"

#include "NeuroData/reader/DataProvider.h"
#include "layers/AbstractLayerEngine.h"

// 이 기능 땜에 NeuroKernel에 NeuroData library를 참조하게 생겼다.
// 나중에 이걸 수정할지 아님 이대로 사용할지 결정해야 한다.
namespace np
{
	namespace engine
	{
		struct _PRODUCER_LAYER_DATA_SET
		{
			_PRODUCER_LAYER_DATA_SET()
			{
				producer = NULL;
				producer_dim_size = 0;
				read_label = false;

				layer_mm = NULL;
				layer_buffer = NULL;
				layer_data_size = 0;
			}
			dp::preprocessor::AbstractProducer* producer;
			neuro_u32 producer_dim_size;
			bool read_label;

			const core::MemoryManager* layer_mm;
			void* layer_buffer;
			neuro_u32 layer_data_size;
		};
		typedef std::vector<_PRODUCER_LAYER_DATA_SET> _producer_layer_data_vector;

		class MiniBatchGenerator
		{
		public:
			MiniBatchGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector);
			virtual ~MiniBatchGenerator();

			neuro_u32 GetBatchSize() const{ return m_batch_size; }
			neuro_size_t GetBatchCount() const {
				return m_batch_count;
			}
			neuro_size_t GetTotalDataCount() const{
				return m_total_data_count;
			}

			virtual bool Ready(neuro_u32 batch_size);

			virtual bool NewEpochStart(){
				return true;
			}

			/*
			if return false, it is eof
			*/
			virtual bool NextBatch() {
				return true;
			}

			/*
				if return false, it is failed
			*/
			neuro_u32 ReadBatchData(bool is_learn_mode);

		protected:
			virtual neuro_u32 ReadBatchDataFromProvider(bool is_learn_mode);

			virtual inline neuro_size_t GetRelativePosition(neuro_size_t pos) const { return pos; }

			neuro_u32 m_batch_size;
			neuro_size_t m_batch_count;

			neuro_size_t m_position;
			neuro_size_t m_total_data_count;

			bool m_is_sequence_read;	// for time series

			dp::preprocessor::DataProvider& m_provider;

			const _producer_layer_data_vector& m_data_vector;

			_TYPED_DATA_VECTOR<void*, 4> m_cpu_read_buffer;
		};

		class MiniBatchSequentialGenerator : public MiniBatchGenerator
		{
		public:
			MiniBatchSequentialGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector);
			virtual ~MiniBatchSequentialGenerator();
		};

		class MiniBatchShuffleGenerator : public MiniBatchSequentialGenerator
		{
		public:
			MiniBatchShuffleGenerator(dp::preprocessor::DataProvider& provider, const _producer_layer_data_vector& data_vector);
			virtual ~MiniBatchShuffleGenerator();

			bool Ready(neuro_u32 batch_size) override;

		protected:
			virtual inline neuro_size_t GetRelativePosition(neuro_size_t pos) const { return m_permutation[pos]; }

			std::vector<neuro_size_t> m_permutation;
		};
	}
}

