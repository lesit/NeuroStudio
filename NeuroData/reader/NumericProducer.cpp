#include "stdafx.h"

#include "NumericProducer.h"

#include "TextReader.h"

using namespace np;
using namespace np::dp;
using namespace np::dp::preprocessor;

NumericProducer::NumericProducer(const model::AbstractProducerModel& model)
: AbstractProducer(model)
, m_model((const model::NumericProducerModel&)model)
, m_isOneHot(m_model.GetLabelOutCount() >= 2)
, m_using_colums(m_model.GetUsingSourceColumnVector())
{
	m_data_count = 0;
	m_position = 0;

	if (m_using_colums.ma_count>0)
	{
		m_last_prev_sum_values.Alloc(m_using_colums.ma_count);
		m_prev_value_circular_table.Alloc(m_last_prev_sum_values.count * m_using_colums.max_ma);
	}
}

bool NumericProducer::Create(DataReaderSet& reader_set)
{
	if (m_using_colums.ma_count == 0)
		return false;

	if (m_model.GetInput() == NULL || m_model.GetModelType() != model::_model_type::reader)
		return false;

	m_reader = reader_set.GetReader(m_model.GetInput()->uid);
	if (m_reader == NULL)
		return false;

	m_reader_type = ((model::AbstractReaderModel*)m_model.GetInput())->GetReaderType();

	m_data_count = m_reader->GetDataCount();
	return true;
}

NumericProducer::~NumericProducer()
{
	m_last_prev_sum_values.Dealloc();
	m_prev_value_circular_table.Dealloc();
}

void NumericProducer::OnPreloadCompleted()
{
}

neuro_size_t NumericProducer::GetRawDataCount() const
{
	return  m_data_count;
}

bool NumericProducer::ReadMaPrevData(neuro_size_t pos)
{
	if (pos<m_using_colums.max_ma-1)
	{
		DEBUG_OUTPUT(L"not available position[%llu], available[%llu]", pos, m_using_colums.max_ma - 1);
		return false;
	}

	memset(m_last_prev_sum_values.buffer, 0, m_last_prev_sum_values.count*sizeof(neuron_value));

	// 순차적으로 읽는 상태가 아닌경우는 MV에 대해 sum을 다시 계산해야 하기 때문에
	const neuro_u32 column_count = m_isOneHot ? 1 : m_using_colums.column_vector.size();

	neuron_value* prev_table = &m_prev_value_circular_table.buffer[0];
	// ReadData에서 앞에것은 버릴것이기 때문에 처음엔 0을 입력한다.
	for (neuro_u32 table_index = 0; table_index < m_using_colums.ma_count; table_index++)
		prev_table[table_index] = 0;

	pos += 1 - m_using_colums.max_ma;

	// 버릴 값들을 위에서 처리했기 때문에 -1 만큼만 한다.
	neuro_u32 count = m_using_colums.max_ma - 1;
	for (neuro_u32 index = 0; index < count; index++, pos++)
	{
		if (!m_reader->Read(pos))	// read는 max_ma -1 만큼만 하면 된다.
		{
			DEBUG_OUTPUT(L"failed to read from filter");
			return false;
		}

		neuro_u32 ma_seq = m_using_colums.max_ma - index;

		prev_table += m_using_colums.ma_count;

		neuro_u32 table_index = 0;
		for (size_t column_index = 0; column_index < column_count; column_index++)
		{
			const model::_NUMERIC_SOURCE_COLUMN& column_info = m_using_colums.column_vector[column_index];
			if (column_info.ma <= 1)
				continue;

			neuron_value value;
			if (ma_seq<= column_info.ma)	// 범위에 있는 값만 적용시켜야 한다.
				value = GetData(column_index);
			else
				value = 0;

			m_last_prev_sum_values.buffer[table_index] += value;
			prev_table[table_index] = value;

			table_index++;
		}
	}

	m_circular_first_index = 0;

	return true;
}

neuro_u32 NumericProducer::ReadRawData(neuro_size_t pos, neuron_value* buffer, bool is_ndf_mode)
{
	if (!m_reader)
		return 0;

	if (m_using_colums.column_vector.size() == 0)
		return 0;

	if (m_using_colums.max_ma > 1 && m_position+1!=pos)	// ma를 구해야 하는데, 혹시라도 순서가 바뀌었다면 이전 ma값들을 구해야 한다.
	{
		if(!ReadMaPrevData(pos))
			return 0;
	}

#ifdef _DEBUG
	if (pos == 123)
		int a = 0;
#endif
	if (!m_reader->Read(pos))
		return 0;
	m_position = pos;

	neuron_value* new_circular_table_value = &m_prev_value_circular_table.buffer[m_circular_first_index * m_using_colums.ma_count];

	neuro_u32 table_index = 0;
	for (neuro_u32 column_index = 0; column_index<m_data_dim_size; column_index++)
	{
		const model::_NUMERIC_SOURCE_COLUMN& column_info = m_using_colums.column_vector[column_index];
		/*	MovePosition 에서 검증했으니까 더는 하지 말자!
		if (column_info.src_column >= m_read_buffer.count)
			return 0;
			*/
		if (column_info.ma > 1)
		{
			neuro_u32 circular_first_index = (m_circular_first_index + m_using_colums.max_ma - column_info.ma) % m_using_colums.max_ma;

			neuron_value* prev_value_ptr = &m_prev_value_circular_table.buffer[circular_first_index * m_using_colums.ma_count];
			m_last_prev_sum_values.buffer[table_index] -= prev_value_ptr[table_index];
			m_last_prev_sum_values.buffer[table_index] += GetData(column_index);

			buffer[column_index] = m_last_prev_sum_values.buffer[table_index] / (neuro_float)column_info.ma;

			new_circular_table_value[table_index] = GetData(column_index);

			++table_index;
		}
		else
			buffer[column_index] = GetData(column_index);

//		value[column_index] /= scale;
	}
	m_circular_first_index = (m_circular_first_index + 1) % m_using_colums.max_ma;;

	return m_data_dim_size;
}

bool NumericProducer::ReadRawLabel(neuro_size_t pos, neuro_u32& label)
{
	if (pos >= m_data_count)
		return false;

	neuro_float value;
	if (!ReadRawData(pos, &value))
		return false;

	label = neuro_u32(value);
	return true;
}

#include "util/StringDataFormat.h"
inline neuron_value NumericProducer::GetData(neuro_u32 column)
{
	if (m_reader_type == model::_reader_type::text)
	{
		const std::string* text = ((TextReader*)m_reader)->GetReadText(m_using_colums.column_vector[column].index);
		if (text == NULL)
			return 0;

		return StringTransform(_data_type::float32, text->c_str());
	}
	else
	{

	}
	return 0;
}
