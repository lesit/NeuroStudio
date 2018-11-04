#pragma once

#include "AbstractReader.h"

#include "model/BinaryReaderModel.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class BinaryReader : public AbstractReader
			{
			public:
				BinaryReader(const model::AbstractReaderModel& model);
				virtual ~BinaryReader();

				bool Create(DataReaderSet& reader_set) override;

				neuro_u64 GetDataCount() const override{ return m_data_count; }
				bool Read(neuro_size_t pos) override;

			private:
				const model::BinaryReaderModel& m_model;
				
				neuro_u64 m_data_count;
				neuro_size_t m_position;
			};
		}
	}
}

