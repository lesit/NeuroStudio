#pragma once

#include "common.h"
#include "AbstractProducer.h"
#include "model/ImageFileProducerModel.h"

#include "gui/Win32/Win32Image.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class ImageFileProducer : public AbstractProducer
			{
			public:
				ImageFileProducer(const model::AbstractProducerModel& model);
				virtual ~ImageFileProducer();

				bool Create(DataReaderSet& reader_set) override;

				neuro_size_t GetRawDataCount() const override;
				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode) override;

				bool ReadRawLabel(neuro_size_t pos, neuro_u32& label) override;

			private:
				const model::ImageFileProducerModel& m_model;

				DataSourceNameVector<wchar_t> m_source_vector;

				gui::win32::ReadImage m_img;
			};
		}
	}
}
