#pragma once

#include "common.h"

#include "AbstractReader.h"

#include "model/TextReaderModel.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			class TextReader : public AbstractReader
			{
			public:
				TextReader(const dp::model::AbstractReaderModel& model);
				virtual ~TextReader();

				bool Create(DataReaderSet& reader_set) override;

				inline neuro_size_t GetDataCount() const
				{
					return m_content_vector.size();
				}

				bool Read(neuro_size_t pos);

				inline const std::string* GetReadText(neuro_u32 src_column) const
				{
					if (m_position >= m_content_vector.size())
						return NULL;

					for (neuro_u32 i = 0; i < m_using_index_vector.size(); i++)
					{
						if (src_column == m_using_index_vector[i])
							return &m_content_vector[m_position][i];
					}
					return NULL;
				}
			protected:
				const dp::model::TextReaderModel& m_model;

				_std_u32_vector m_using_index_vector;

				std::vector<std_string_vector> m_content_vector;
				neuro_size_t m_position;
			};
		}
	}
}
