#pragma once

#include "common.h"
#include "storage/DeviceAdaptor.h"

#include "reader/TextReader.h"

namespace np
{
	namespace dp
	{
		struct _STREAM_WRITE_ROW_INFO
		{
			enum class _source_type{ no, value, ret_text_source };
			_STREAM_WRITE_ROW_INFO()
			{
				type = _source_type::value;

				value_onehot = false;
				value_index = neuro_last32;

				ref_text_source_index = 0;
			}

			_source_type type;

			bool value_onehot;
			neuro_u32 value_index;	// value_onehot이 아닐때 value값중 출력값. neuro_last32이면 모두다 출력

			neuro_u32 ref_text_source_index;
		};
		typedef std::vector<_STREAM_WRITE_ROW_INFO> _stream_write_row_vector;

		struct _STREAM_WRITE_INFO
		{
			_STREAM_WRITE_INFO()
			{
				device = NULL;
				row_delimiter = ",";
				col_delimiter = "\n";

				no_start = 0;

				no_length = 5;

				value_float_length = 0;
				value_float_under_length = 7;
			}
			_STREAM_WRITE_INFO(const _STREAM_WRITE_INFO& src)
			{
				*this = src;
			}
			_STREAM_WRITE_INFO& operator =(const _STREAM_WRITE_INFO& src)
			{
				device = src.device;

				no_type_prefix = src.no_type_prefix;
				no_start = src.no_start;

				row_delimiter = src.row_delimiter;
				col_delimiter = src.col_delimiter;

				col_vector = src.col_vector;
				return *this;
			}

			device::DeviceAdaptor* device;

			neuro_u32 no_start;

			std::string no_type_prefix;
			neuro_u32 no_length;

			neuro_u32 value_float_length;
			neuro_u32 value_float_under_length;

			std::string row_delimiter;
			std::string col_delimiter;

			_stream_write_row_vector col_vector;
		};
		class StreamWriter
		{
		public:
			StreamWriter(const _STREAM_WRITE_INFO& write_info, preprocessor::TextReader* ref_text_source);
			virtual ~StreamWriter();

			bool Write(const _NEURO_TENSOR_DATA& tensor_data);
			bool Write(const _VALUE_VECTOR& value_source);

		private:
			bool WriteSample(const _VALUE_VECTOR& value_source);
			bool WriteColumnDelimeter();

			_STREAM_WRITE_INFO m_write_info;

			preprocessor::TextReader* m_ref_text_source;	// 역시 DocParser는 소스로 들어가야 한다
			_std_u32_vector m_ret_text_index_vector;

			neuro_u32 m_position;
		};
	}
}
