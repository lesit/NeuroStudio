#pragma once

#include "AbstractProducer.h"
#include "model/MnistProducerModel.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class MnistProducer : public AbstractProducer
			{
			public:
				virtual ~MnistProducer();

				neuro_size_t GetRawDataCount() const override {	return m_data_count;}
			protected:
				MnistProducer(const model::AbstractProducerModel& model);

				void OnPreloadCompleted() override;

				neuro_u8* m_temp_read_buffer;

				neuro_u32 m_data_count;
			};

			class MnistImageProducer : public MnistProducer
			{
			public:
				MnistImageProducer(const model::AbstractProducerModel& model);
				virtual ~MnistImageProducer();

				virtual const wchar_t* GetTypeString() const { return L"Mnist image"; }

				bool Create(DataReaderSet& reader_set) override;

				virtual bool SupportPreload() const { return true; }

				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode = false) override;

			protected:
				const model::MnistImageProducerModel& m_model;

				NP_SIZE m_img_sz;
			};

			class MnistLabelProducer : public MnistProducer
			{
			public:
				MnistLabelProducer(const model::AbstractProducerModel& model);
				virtual ~MnistLabelProducer();

				virtual const wchar_t* GetTypeString() const { return L"Mnist label"; }

				bool Create(DataReaderSet& reader_set) override;

				// mnist label은 이미 m_temp_read_buffer에 저장해놓고 있기 때문에 preload할 필요가 없다.
				bool ReadRawLabel(neuro_size_t pos, neuro_u32& label) override;
				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode = false) override { return 0; }

			protected:
				const model::MnistLabelProducerModel& m_model;
			};
		}
	}
}
