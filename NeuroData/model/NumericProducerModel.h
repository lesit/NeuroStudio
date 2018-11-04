#pragma once

#include "common.h"
#include "AbstractProducerModel.h"
#include "TextReaderModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			struct _NUMERIC_SOURCE_COLUMN
			{
				_NUMERIC_SOURCE_COLUMN()
				{
					index = 0;
					ma = 1;
				}
				_NUMERIC_SOURCE_COLUMN(neuro_u32 index, neuro_u32 ma = 1)
				{
					this->index = index;
					this->ma = ma;
				}

				neuro_u32 index;	// data filter에서의 index. 최소 0이상
				neuro_u32 ma;		// moving average. 최소 1이상
			};
			typedef std::vector<_NUMERIC_SOURCE_COLUMN> _numeric_source_column_vector;

			struct _NUMERIC_USING_SOURCE_COLUMNS
			{
				_NUMERIC_USING_SOURCE_COLUMNS()
				{
					max_ma = ma_count = 0;
				}
				_NUMERIC_USING_SOURCE_COLUMNS& operator=(const _NUMERIC_USING_SOURCE_COLUMNS& src)
				{
					column_vector = src.column_vector;
					max_ma = src.max_ma;
					ma_count = src.ma_count;
				}
				_numeric_source_column_vector column_vector;
				neuro_u32 max_ma;
				neuro_u32 ma_count;
			};

			class NumericProducerModel : public AbstractProducerModel
			{
			public:
				NumericProducerModel(DataProviderModel& provider, neuro_u32 uid);
				virtual ~NumericProducerModel();

				_producer_type GetProducerType() const override { return _producer_type::numeric; }

				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::must; }

				tensor::DataShape GetDataShape() const override;

				_label_out_type GetLabelOutType() const override { return _label_out_type::direct_def; }
				neuro_u32 GetLabelOutCount() const override  {
					return m_onehot_encoding_size;
				}
				void SetLabelOutCount(neuro_u32 scope) override  
				{
					m_onehot_encoding_size = scope; 
					if (scope > 1 && !m_using_colum_map.empty())
					{
						std::map<neuro_u32, neuro_u32>::const_iterator it = m_using_colum_map.begin();
						neuro_u32 column = it->first;
						neuro_u32 ma = it->second;
						m_using_colum_map.clear();
						m_using_colum_map[column] = ma;
					}
				}

				std::string MakeNdfPath(const std::string& source_name) const override;

				bool SupportNdfClone() const override { return true; }
				_ndf_dim_type GetNdfDimType() const override;

				void ChangedProperty() override
				{
					if (GetInput())
					{
						neuro_u32 max_column_count = GetInput()->GetColumnCount();
						std::map<neuro_u32, neuro_u32>::const_reverse_iterator it = m_using_colum_map.rbegin();
						for (; it != m_using_colum_map.rend(); it++)
						{
							if (it->first >= max_column_count)
								m_using_colum_map.erase(it->first);
							else
								break;
						}
					}
					else
					{
						m_using_colum_map.clear();
					}
					__super::ChangedProperty();
				}

				void InsertSourceColumn(neuro_u32 column, neuro_u32 ma = 1) {
					if(m_onehot_encoding_size>0)
						m_using_colum_map.clear();

					m_using_colum_map[column] = ma;
					ChangedProperty();
				}
				void EraseSourceColumn(neuro_u32 column) {
					m_using_colum_map.erase(column);
					ChangedProperty();
				}

				const std::map<neuro_u32, neuro_u32>& GetUsingSourceColumns() const { return m_using_colum_map; }

				void GetUsingSourceIndexSet(_u32_set& index_set) const override
				{
					std::map<neuro_u32, neuro_u32>::const_iterator it = m_using_colum_map.begin();
					for(;it!= m_using_colum_map.end();it++)
						index_set.insert(it->first);
				}

				_NUMERIC_USING_SOURCE_COLUMNS GetUsingSourceColumnVector() const;
			protected:
				neuro_u32 GetAvailableStartPosition() const override;

			private:
				neuro_u32 m_onehot_encoding_size;

				std::map<neuro_u32, neuro_u32> m_using_colum_map;
			};
		}
	}
}
