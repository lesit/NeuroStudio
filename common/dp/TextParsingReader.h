#pragma once

#include <string>
#include <unordered_map>

#include "../storage/FileDeviceAdaptor.h"
#include "TextInputStream.h"

namespace np
{
	namespace dp
	{
		enum class _delimiter_type{ token, length};

		typedef std::vector<std::pair<std::string, std::string>> _change_vector;
		class TextColumnDelimiter
		{
		public:
			TextColumnDelimiter()
			{
				change_vector.push_back({ "\"\"", " " });
				change_vector.push_back({ "\r", "" });
			}
			virtual ~TextColumnDelimiter() {}

			virtual dp::_delimiter_type GetType() const = 0;

			TextColumnDelimiter& operator = (const TextColumnDelimiter& src)
			{
				change_vector = src.change_vector;
				return *this;
			}
			_change_vector change_vector;
		};

		class TextColumnLengthDelimiter : public TextColumnDelimiter
		{
		public:
			dp::_delimiter_type GetType() const { return dp::_delimiter_type::length; }

			TextColumnLengthDelimiter& operator = (const TextColumnLengthDelimiter& src)
			{
				((TextColumnDelimiter&)*this) = src;
				lengh_vector = src.lengh_vector;
				return *this;
			}

			std::vector<neuro_u32> lengh_vector;
		};

		class TextColumnTokenDelimiter : public TextColumnDelimiter
		{
		public:
			TextColumnTokenDelimiter()
			{
				content_token = "\n";

				column_token_vector = { ",", "\t" };
				double_quote = true;
			}
			dp::_delimiter_type GetType() const { return dp::_delimiter_type::token; }

			TextColumnTokenDelimiter& operator = (const TextColumnTokenDelimiter& src)
			{
				((TextColumnDelimiter&)*this) = src;
				content_token = src.content_token;
				column_token_vector = src.column_token_vector;
				double_quote = src.double_quote;
				return *this;
			}

			std::string content_token;

			std_string_vector column_token_vector;
			bool double_quote;
		};

		class TextParsingReader
		{
		public:
			static TextParsingReader* CreateInstance(const TextColumnDelimiter& delimiter, const _std_u32_vector* index_vector=NULL);

			virtual ~TextParsingReader();

			bool SetInputDevice(device::DeviceAdaptor& input_device);
			bool SetFile(const char* file_path);

			neuro_u64 ReadAllContents(neuro_u32 skip_count, bool reverse, std::vector<std_string_vector>& content_vector);

			neuro_u64 CalculateContentCount();
			const std::vector<const char*>& GetLinePosVector() const { return m_line_pos_vector; }

			bool MoveContentPosition(neuro_u64 pos);
			bool ReadContent(std_string_vector* content, bool& eof);

		protected:
			TextParsingReader(const _std_u32_vector* index_vector);

			virtual neuro_u32 GetColumnCount() const = 0;

			virtual neuro_size_t ReadText(neuro_u32 column, std::string* ret_text, bool& end_of_content) = 0;

			_std_u32_vector m_index_vector;

			device::FileDeviceAdaptor m_fda;

			char* m_text_buffer;
			const char* m_text_read_cur;
			const char* m_text_read_last;

			std::vector<const char*> m_line_pos_vector;
			const bool m_calc_count_comleted;
		};

		class FixedlenContentReader : public TextParsingReader
		{
		public:
			FixedlenContentReader(TextColumnLengthDelimiter& delimiter, const _std_u32_vector* index_vector);

			neuro_size_t ReadText(neuro_u32 column, std::string* ret_text, bool& end_of_content) override;

		private:
			neuro_u32 GetColumnCount() const override { return m_delimiter.lengh_vector.size(); }

			const TextColumnLengthDelimiter& m_delimiter;
		};

		class TokenContentReader : public TextParsingReader
		{
		public:
			TokenContentReader(TextColumnTokenDelimiter& delimiter, const _std_u32_vector* index_vector);

			neuro_size_t ReadText(neuro_u32 column, std::string* ret_text, bool& end_of_content) override;

		private:
			TextColumnTokenDelimiter& m_delimiter;

			neuro_u32 GetColumnCount() const override { return neuro_last32; }

			std::vector<const char*> token_vector;

			struct _FIND_TOKEN
			{
				_FIND_TOKEN()
				{
					str = change = NULL;
					len = change_len = 0;
				}
				_FIND_TOKEN(const char* _str, const char* _change = NULL)
				{
					str = _str;
					len = _str ? strlen(str) : 0;
					change = _change;
					if (change)
						change_len = change ? strlen(change) : 0;
				}
				const char* str;
				size_t len;
				const char* change;
				size_t change_len;
			};

			typedef std::vector<_FIND_TOKEN> _find_token_vector;
			neuro_size_t ReadStringUntilToken(const _find_token_vector& find_token_vector, std::string& found_token, std::string* ret_text);

			const _FIND_TOKEN* FindToken(const _find_token_vector& find_token_vector, const char* str, size_t len);

			static int strlen_compare(const void* a, const void* b);
			void CreateFindTokenVector(const std::vector<const char*> &token_vector, const _change_vector& change_vector, _find_token_vector& find_token_vector);

			_find_token_vector m_first_find_token_vector;
			_find_token_vector m_end_quote_find_token_vector;
		};

	}
}
