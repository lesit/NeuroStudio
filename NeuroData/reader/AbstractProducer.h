#pragma once

#include "AbstractPreprocessor.h"

#include "../model/AbstractProducerModel.h"

#include "storage/DeviceAdaptor.h"
#include "DataReaderSet.h"

#include "DataPreloadBuffer.h"

namespace np
{
	namespace dp
	{
		// 나중에 실제로 Data Stream 처리기가 분리될때 각 함수를 export 시킬수 있어야 한다.
		// 즉, Neuro Studio에서 이 kernel을 호출시키는 관점으로 봐야 한다.
		// 따라서, kernel에 이 모든것을 다 담던지, 아니면 stream 처리기를 다른 모듈로 분리해서

		namespace preprocessor
		{
			class MemProducer;
			class AbstractProducer : public AbstractPreprocessor
			{
			public:
				static AbstractProducer* CreateInstance(const char* status
					, DataReaderSet& reader_set
					, const dp::model::AbstractProducerModel& model);

				virtual ~AbstractProducer();

				model::_model_type GetModelType() const override { return model::_model_type::producer; }

				virtual const wchar_t* GetTypeString() const { return L"unknown"; }

				virtual bool SupportShuffle() const { return true; }

				virtual bool Preload();

				virtual neuro_size_t GetDataCount() const { return GetRawDataCount() > m_start ? GetRawDataCount() - m_start : 0; }
				bool Read(neuro_size_t pos, neuron_value* buffer);
				virtual neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* buffer, bool is_ndf_mode = false) = 0;

				bool ReadLabel(neuro_size_t pos, neuro_u32& label);
				virtual bool ReadRawLabel(neuro_size_t pos, neuro_u32& label) { return 0; }

				MemProducer* CreateCloneMemoryProducer() const;

				const tensor::DataShape m_data_shape;
				const neuro_u32 m_data_dim_size;

				const neuron_value m_scale_min;
				const neuron_value m_scale_max;

				const neuro_size_t m_start;

				const model::_label_out_type m_label_out_type;

				// for data provider
				bool data_noising;

#ifdef _DEBUG
			public:
#endif
			protected:
				AbstractProducer(const tensor::DataShape& data_shape
					, neuro_size_t start = 0
					, neuron_value scale_min = -1.f, neuron_value scale_max = 1.f
					, model::_label_out_type label_out_type = model::_label_out_type::none);

				AbstractProducer(const model::AbstractProducerModel& model);

				virtual neuro_size_t GetRawDataCount() const = 0;

				bool PreloadStart(neuro_size_t start_pos);

				virtual void DataNoising(neuron_value* value) {}

				virtual bool SupportPreload() const { return false; }

				virtual void OnPreloadCompleted() {}

				void SetPadding(neuron_value* ptr, neuro_u32 size) const;

				DataPreloadBuffer m_preload_buffer;
				DataPreloadBuffer m_label_preload_buffer;
			};

			class MemProducer : public AbstractProducer
			{
			public:
				MemProducer(neuro_u64 total_size, neuro_u32 row_size = neuro_last32);
				MemProducer(const _VALUE_VECTOR& buf, neuro_u32 row_size = neuro_last32);

				virtual ~MemProducer();

				bool Create(DataReaderSet& reader_set) override { return false; }

				bool SetData(neuron_value* value, neuro_u32 size);

				neuro_u64 GetRawDataCount() const override {
					return m_data_count;
				}

				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode) override;

				_VALUE_VECTOR m_value_buffer;

			protected:
				bool SupportPreload() const override { return false; }

			private:
				neuro_u64 m_data_count;
				bool m_clone_buffer;
			};
		}
	}
}
