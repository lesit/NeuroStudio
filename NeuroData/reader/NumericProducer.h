#pragma once

#include "common.h"
#include "AbstractProducer.h"

#include "model/NumericProducerModel.h"
#include "model/AbstractReaderModel.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class NumericProducer : public AbstractProducer
			{
			public:
				NumericProducer(const model::AbstractProducerModel& model);
				virtual ~NumericProducer();

				bool Create(DataReaderSet& reader_set) override;

				virtual const wchar_t* GetTypeString() const { return L"NumericProducerModel"; }

				bool SupportShuffle() const { return m_using_colums.ma_count==0; }

				bool ReadRawLabel(neuro_size_t pos, neuro_u32& label) override;

				virtual neuro_size_t GetRawDataCount() const override;
				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode = false) override;
			protected:

				virtual void OnPreloadCompleted() override;

				bool ReadMaPrevData(neuro_u64 pos);

				neuron_value GetData(neuro_u32 column);

			private:
				const model::NumericProducerModel& m_model;

				const bool m_isOneHot;

				const model::_NUMERIC_USING_SOURCE_COLUMNS m_using_colums;

				AbstractReader* m_reader;
				model::_reader_type m_reader_type;

				neuro_size_t m_data_count;

				neuro_size_t m_position;

				_VALUE_VECTOR m_last_prev_sum_values;
				_VALUE_VECTOR m_prev_value_circular_table;
				neuro_u32 m_circular_first_index;
			};
		}
	}
}
