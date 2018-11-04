#pragma once

#include "AbstractReaderModel.h"

#include "dp/TextParsingReader.h"

#include "util/StringUtil.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			static const wchar_t* _delimiter_type_string[] = { L"Token", L"Length" };

			static dp::_delimiter_type ToDelimiterType(const wchar_t* name)
			{
				for (neuro_u8 type = 0; type < _countof(_delimiter_type_string); type++)
				if (wcscmp(name, _delimiter_type_string[type]) == 0)
					return (dp::_delimiter_type)type;

				return dp::_delimiter_type::token;
			}

			static std::string TokenVectorToString(const std_string_vector& token_vector)
			{
				std::string token;
				if (token_vector.size() == 1)
					token = util::StringUtil::TransformAsciiToText(token_vector[0]);
				else if (token_vector.size() > 1)
				{
					for (size_t i = 0, n = token_vector.size(); i < n; i++)
					{
						token += '[';
						token.append(util::StringUtil::TransformAsciiToText(token_vector[i]));
						token += ']';
					}
				}
				return token;
			}

			static void StringToTokenVector(const std::string& str, std_string_vector& token_vector)
			{
				token_vector.clear();
				if (str.size() > 0)
				{
					if (str[0] == '[')
					{
						int start = 0;
						int end;
						while (start >= 0 && (end = str.find(']', start + 1)) != std::wstring::npos)
						{
							std::string token(str.begin() + start + 1, str.begin() + end);
							token = util::StringUtil::TransformTextToAscii(token);
							token_vector.push_back(token);

							start = str.find('[', end + 1);
						}
					}
					else
						token_vector.push_back(util::StringUtil::TransformTextToAscii(str));
				}
			}

			class TextReaderModel : public AbstractReaderModel
			{
			public:
				TextReaderModel(DataProviderModel& provider, neuro_u32 uid);
				virtual ~TextReaderModel();

				_input_source_type GetInputSourceType() const override {
					if (m_input)
						return _input_source_type::none;

					return _input_source_type::textfile;
				}
				_reader_type GetReaderType() const override{ return _reader_type::text; }

				bool CopyFrom(const AbstractReaderModel& src) override
				{
					if (!__super::CopyFrom(src))
						return false;

					TextReaderModel& src_reader = (TextReaderModel&)src;
					m_imported_source = src_reader.m_imported_source;

					m_skip_count = src_reader.m_skip_count;
					m_is_reverse = src_reader.m_is_reverse;

					m_column_delimiter = src_reader.m_column_delimiter;
					return true;
				}

				void SetImportedSource(const char* name) { m_imported_source = name; }
				const char* GetImportedSource() const { return m_imported_source.c_str(); }
				bool IsImported() const { return !m_imported_source.empty(); }

				void SetSkipFirstCount(neuro_u32 skip_count){ m_skip_count = skip_count; }
				neuro_u32 GetSkipFirstCount() const { return m_skip_count; }

				void SetReverse(bool reverse){ m_is_reverse = reverse; }
				bool IsReverse() const{ return m_is_reverse; }

				void ChangeDelimiterType(_delimiter_type type);
				TextColumnDelimiter& GetDelimiter() { return *m_column_delimiter;}
				const TextColumnDelimiter& GetDelimiter() const { return *m_column_delimiter;}

/*
				std::string TextReaderModel::GetColumnDelimiterTokensString() const
				{
					if (m_column_delimiter.type == dp::_delimiter_type::token)
						return TokenVectorToString(m_column_delimiter.token_vector);
					return "";
				}
				*/

				void SetColumnCount(neuro_u32 count) override;
				neuro_u32 GetColumnCount() const override;

				bool ImportCSV(device::FileDeviceAdaptor& input_device);
				bool Import(device::FileDeviceAdaptor& input_device, const  TextColumnTokenDelimiter& default_filter);

			private:
				//nlp::_content_col_delimiter_vector::iterator FindColumn(dp::_CONTENT_COL_DELIMITER* column);

				std::string m_imported_source;

				neuro_u32 m_skip_count;
				bool m_is_reverse;

				TextColumnDelimiter* m_column_delimiter;
			};

			class ExtTextColumnTokenDelimiter : public TextColumnTokenDelimiter
			{
			public:
				ExtTextColumnTokenDelimiter()
				{
					m_column_count = 0;
				}

				neuro_u32 m_column_count;
			};
		}
	}
}
