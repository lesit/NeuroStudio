#include "stdafx.h"
#include "StreamWriter.h"

#include "util/StringUtil.h"

using namespace np;
using namespace np::dp;

StreamWriter::StreamWriter(const _STREAM_WRITE_INFO& write_info, preprocessor::TextReader* ref_text_source)
{
	m_write_info = write_info;
	m_ref_text_source = ref_text_source;

	m_position = 0;

	for (neuro_size_t i = 0, n = m_write_info.col_vector.size(); i < n; i++)
	{
		const _STREAM_WRITE_ROW_INFO& info = m_write_info.col_vector[i];

		if (info.type == _STREAM_WRITE_ROW_INFO::_source_type::ret_text_source)
			m_ret_text_index_vector.push_back(info.ref_text_source_index);
	}
}

StreamWriter::~StreamWriter()
{
	if (m_write_info.device)
		delete m_write_info.device;
}

bool StreamWriter::Write(const _NEURO_TENSOR_DATA& tensor_data)
{
	if (m_write_info.col_vector.size() == 0)
		return true;

	for (neuro_u32 i = 0, n = tensor_data.GetBatchSize(); i < n; i++)
	{
		_VALUE_VECTOR value=tensor_data.GetSample(i);
		if (!Write(value))
			return false;
	}
	return true;
}

bool StreamWriter::Write(const _VALUE_VECTOR& value)
{
	if (m_write_info.col_vector.size() == 0)
		return true;

	if(!WriteSample(value))
		return false;

	return WriteColumnDelimeter();
}

bool StreamWriter::WriteSample(const _VALUE_VECTOR& value_source)
{
	if (!m_write_info.device)
	{
		DEBUG_OUTPUT(L"no writable device");
		return false;
	}

	std::vector<std::string> text_vector;
	if (m_ref_text_source)
	{
		m_ref_text_source->Read(m_position);

		for (neuro_u32 i = 0; i < m_ret_text_index_vector.size(); i++)
		{
			const std::string* text = m_ref_text_source->GetReadText(m_ret_text_index_vector[i]);
			if (!text)
			{
				DEBUG_OUTPUT(L"no text");
				return false;
			}
			text_vector.push_back(*text);
		}
	}

	std::string str;
	for (neuro_size_t i = 0, text_index = 0, n = m_write_info.col_vector.size(); i < n; i++, text_index++)
	{
		if (i>0)
			str.append(m_write_info.row_delimiter);

		const _STREAM_WRITE_ROW_INFO& info = m_write_info.col_vector[i];

		std::string column;
		if (info.type == _STREAM_WRITE_ROW_INFO::_source_type::ret_text_source)
		{
			column = text_vector[text_index];
		}
		else
		{
			std::string format = "%";
			if (info.type == _STREAM_WRITE_ROW_INFO::_source_type::no)
			{
				column += m_write_info.no_type_prefix;
				format += "0";
				format += util::StringUtil::Format<char>("%u", m_write_info.no_length);
				format.append("u");
			}
			else if (info.type == _STREAM_WRITE_ROW_INFO::_source_type::value)
			{
				format += util::StringUtil::Format<char>("%u.%u", m_write_info.value_float_length, m_write_info.value_float_under_length);
				format.append("f");
			}

			if (info.type == _STREAM_WRITE_ROW_INFO::_source_type::no)
			{
				column += util::StringUtil::Format<char>(format.c_str(), m_write_info.no_start + m_position);
			}
			else
			{
				if (info.value_onehot)
				{
				}
				else
				{
					if (info.value_index == neuro_last32)
					{
					}
					else
					{
						neuro_float value = value_source.buffer[info.value_index];
						if (value >= 0.9999999999f)
							value = 0.9999999999f;
						else if (value <= 0)
							value = 0.0000000001f;

						column = util::StringUtil::Format<char>(format.c_str(), value);
					}
				}
			}
		}

		str.append(column);
	}
	if (m_write_info.device->Write(str.c_str(), str.size()) != str.size())
	{
		DEBUG_OUTPUT(L"failed write");
		return false;
	}

	++m_position;
	return true;
}

bool StreamWriter::WriteColumnDelimeter()
{
	if (m_write_info.device->Write(m_write_info.col_delimiter.c_str(), m_write_info.col_delimiter.size()) != m_write_info.col_delimiter.size())
	{
		DEBUG_OUTPUT(L"failed write delimiter");
		return false;
	}
	return true;
}
