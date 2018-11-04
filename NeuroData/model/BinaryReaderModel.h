#pragma once

#include "AbstractReaderModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			class BinaryReaderModel : public AbstractReaderModel
			{
			public:
				BinaryReaderModel(DataProviderModel& provider, neuro_u32 uid)
					: AbstractReaderModel(provider, uid)
				{
				}
				virtual ~BinaryReaderModel() {}

				_input_source_type GetInputSourceType() const override {
					if (m_input)
						return _input_source_type::none;

					return _input_source_type::binaryfile;
				}

				_reader_type GetReaderType() const override{ return _reader_type::binary; }

				bool CopyFrom(const AbstractReaderModel& src) override
				{
					if (!__super::CopyFrom(src))
						return false;

					m_type_vector = ((BinaryReaderModel&)src).m_type_vector;
					return true;
				}

				void SetColumnCount(neuro_u32 count) override
				{
					m_type_vector.resize(count, _data_type::float32);
					ChangedProperty();
				}
				neuro_u32 GetColumnCount() const override { return m_type_vector.size(); }

				const _data_type_vector& GetTypeVector() const { return m_type_vector; }
				_data_type_vector& GetTypeVector() { return m_type_vector; }
				void SetTypeVector(const _data_type_vector& type_vector)
				{
					m_type_vector = type_vector; 
					ChangedProperty();
				}
			private:
				_data_type_vector m_type_vector;
			};
		}
	}
}
