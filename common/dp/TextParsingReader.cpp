#include "TextParsingReader.h"

#include "util/np_util.h"
#include "util/StringUtil.h"
#include <algorithm>

using namespace np;
using namespace np::util;
using namespace np::dp;

TextParsingReader* TextParsingReader::CreateInstance(const TextColumnDelimiter& delimiter, const _std_u32_vector* index_vector)
{
	if (delimiter.GetType() == _delimiter_type::length)
		return new FixedlenContentReader((TextColumnLengthDelimiter&)delimiter, index_vector);
	else
		return new TokenContentReader((TextColumnTokenDelimiter&)delimiter, index_vector);
}

TextParsingReader::TextParsingReader(const _std_u32_vector* index_vector)
: m_calc_count_comleted(false)
{
	if (index_vector)
		m_index_vector = *index_vector;

	m_text_buffer = NULL;
	m_text_read_cur = m_text_read_last = m_text_buffer;
}

TextParsingReader::~TextParsingReader()
{
	if (m_text_buffer)
		delete[] m_text_buffer;
}

bool TextParsingReader::SetInputDevice(device::DeviceAdaptor& input_device)
{
	m_line_pos_vector.clear();

	TextFileInputStream stream;
	if (!stream.SetInputDevice(&input_device))
	{
		DEBUG_OUTPUT(L"failed TextFileInputStream::SetInputDevice");
		return false;
	}

	DEBUG_OUTPUT(L"start");

	int count;
	m_text_buffer = stream.ReadStringAllUtf8(count);
	m_text_read_cur = m_text_buffer;
	if (m_text_buffer == NULL || count == 0)
	{
		m_text_read_last = m_text_buffer;

		DEBUG_OUTPUT(L"failed read string all");
		return false;
	}

	m_text_read_last = m_text_buffer + count;

	m_line_pos_vector.push_back(m_text_read_cur);

	DEBUG_OUTPUT(L"end");
	return true;
}

bool TextParsingReader::SetFile(const char* file_path)
{
	if (!m_fda.Create(file_path, true, false, false))
		return false;

	return SetInputDevice(m_fda);
}

neuro_u64 TextParsingReader::ReadAllContents(neuro_u32 skip_count, bool reverse, std::vector<std_string_vector>& content_vector)
{
	DEBUG_OUTPUT(L"start");
	if (!m_text_buffer)
	{
		DEBUG_OUTPUT(L"no read buffer");
		return 0;
	}

	m_text_read_cur = m_line_pos_vector[0];

	bool eof = false;

	for (neuro_u32 i = 0; i < skip_count; i++)
	{
		if (!ReadContent(NULL, eof) || eof)
			return 0;
	}
	do
	{
		content_vector.resize(content_vector.size() + 1);

		std_string_vector* content;
		if (reverse)
		{
			content_vector.insert(content_vector.begin(), std_string_vector());
			content = &content_vector.front();
		}
		else
		{
			content_vector.resize(content_vector.size() + 1);
			content = &content_vector.back();
		}
		if (!ReadContent(content, eof))
			break;
	} while (!eof);

	return content_vector.size();
}

neuro_u64 TextParsingReader::CalculateContentCount()
{
	DEBUG_OUTPUT(L"start");
	if (!m_text_buffer)
	{
		DEBUG_OUTPUT(L"no read buffer");
		return 0;
	}

	if (m_calc_count_comleted)
		return m_line_pos_vector.size();

	m_text_read_cur = m_line_pos_vector[0];
	
	bool eof = false;
	while (!eof && ReadContent(NULL, eof))	;

	m_text_read_cur = m_text_buffer;

	MoveContentPosition(0);
	DEBUG_OUTPUT(L"end");
	return m_line_pos_vector.size();
}

bool TextParsingReader::MoveContentPosition(neuro_u64 pos)
{
	if (m_line_pos_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no data");
		return false;
	}

	const neuro_u64 last_index = m_line_pos_vector.size() - 1;
	if (pos == neuro_last64)
	{
		DEBUG_OUTPUT(L"move last");
		m_text_read_cur = m_line_pos_vector[last_index];
		return true;
	}

	if (pos < m_line_pos_vector.size())
	{
		m_text_read_cur = m_line_pos_vector[pos];
		return true;
	}

	DEBUG_OUTPUT(L"not found");

	m_text_read_cur = m_line_pos_vector[last_index];

	bool eof=false;
	neuro_u64 i = 0;
	neuro_u64 n = pos - last_index;	// 만약 last_index=0 일때, pos=0 이면 처음이므로 읽을 필요가 없이 찾은 것이다!
	for (i; !eof && ReadContent(NULL, eof) && i < n; i++)
		;

	return i == n;
}

bool TextParsingReader::ReadContent(std_string_vector* content, bool& eof)
{
	eof = true;

	neuro_size_t total_read = 0;

	neuro_u32 index_vector_index = 0;
	const neuro_u32 index_vector_count = m_index_vector.size();

	neuro_u32 column_count = GetColumnCount();
	for (neuro_u32 column = 0; column<column_count; column++)
	{
#ifdef _DEBUG
		if (column == 3)
			int a = 0;
#endif
		std::string* ret_text = NULL;
		if (content!=NULL
			&& (m_index_vector.size()==0 || index_vector_index < index_vector_count && column == m_index_vector[index_vector_index]))
		{
			content->resize(content->size() + 1);
			ret_text = &content->at(content->size() - 1);

			++index_vector_index;
		}

		bool end_of_content;
		total_read += ReadText(column, ret_text, end_of_content);

		if (ret_text)
		{
			util::StringUtil::Trim(*ret_text);

			if (util::StringUtil::CheckUtf8((unsigned char*)ret_text->c_str(), ret_text->size()) == util::_str_utf8_type::failure)
				DEBUG_OUTPUT(L"failed check utf8 : %s", util::StringUtil::MultiByteToWide(*ret_text).c_str());
		}

		if (m_text_read_cur == m_text_read_last)
		{
			eof = true;
			break;
		}

		if (end_of_content)	// 하나의 content가 끝남. csv에선 하나의 column 완성
			break;
	}

	if (eof)
	{
		const_cast<bool&>(m_calc_count_comleted) = true;

		if (total_read == 0)
		{
			DEBUG_OUTPUT(L"finished");
			return false;
		}
	}
	else
	{
		// eof 인 경우에 다음걸 읽을 필요가 없으므로 다 읽은 m_text_read_cur 를 저장할 필요가 없다.
		// 또한 그래야 m_line_pos_vector 의 크기가 총 데이터 크기가 된다!
		if (m_line_pos_vector.back() < m_text_read_cur)	// 새로 읽는 상태일 경우
			m_line_pos_vector.push_back(m_text_read_cur);
	}
	return true;
}

FixedlenContentReader::FixedlenContentReader(TextColumnLengthDelimiter& delimiter, const _std_u32_vector* index_vector)
	: m_delimiter(delimiter), TextParsingReader(index_vector)
{

}

inline neuro_size_t FixedlenContentReader::ReadText(neuro_u32 column, std::string* ret_text, bool& end_of_content)
{
	end_of_content = false;

	neuro_size_t total_read = 0;

	neuro_u32 read = std::min(m_delimiter.lengh_vector[column], neuro_u32(m_text_read_last - m_text_read_cur));
	if (ret_text)
		ret_text->append(m_text_read_cur, m_text_read_cur + read);
	total_read += read;
	m_text_read_cur += read;
	if (m_delimiter.lengh_vector[column] - read > 0)
	{
		// 더 읽어야 하는데 없는 경우
		m_text_read_cur = m_text_read_last;
		return total_read;
	}
	return total_read;
}

TokenContentReader::TokenContentReader(TextColumnTokenDelimiter& delimiter, const _std_u32_vector* index_vector)
	: m_delimiter(delimiter), TextParsingReader(index_vector)
{
	if (m_delimiter.double_quote)
		token_vector.push_back("\"");
	for (size_t i = 0; i < m_delimiter.column_token_vector.size(); i++)
		token_vector.push_back(m_delimiter.column_token_vector[i].c_str());
	token_vector.push_back(m_delimiter.content_token.c_str());

	CreateFindTokenVector(token_vector, m_delimiter.change_vector, m_first_find_token_vector);
	CreateFindTokenVector({ "\"" }, m_delimiter.change_vector, m_end_quote_find_token_vector);
}

inline neuro_size_t TokenContentReader::ReadText(neuro_u32 column, std::string* ret_text, bool& end_of_content)
{
	end_of_content = false;
	if (ret_text)
		ret_text->clear();

	neuro_size_t total_read = 0;

	while (true)
	{
		std::string found_token;
		total_read += ReadStringUntilToken(m_first_find_token_vector, found_token, ret_text);

		if (m_text_read_cur == m_text_read_last)	// 더이상 읽을게 없다.
			break;

		if (found_token == "\"")	// 다음 하나의 "가 나올때까지.
		{
			// 다음 " 가 나올때까지 무조건 넣는다.
			total_read += ReadStringUntilToken(m_end_quote_find_token_vector, found_token, ret_text);

			if (m_text_read_cur == m_text_read_last)	// 더이상 읽을게 없다.
				break;

			if (found_token != "\"")	// 끝나는 "를 찾지 못하면 끝난거니까 종료한다. 그런데 이미 위에서 eof=true일 것이다.
			{
				DEBUG_OUTPUT(L"no end \"");
				m_text_read_cur = m_text_read_last;
				break;
			}
			continue;
		}

		if (found_token == m_delimiter.content_token)
		{
			end_of_content = true;
			return total_read;// 끝나는 토큰 발견
		}

		for (size_t i = 0; i < m_delimiter.column_token_vector.size(); i++)
		{
			if (found_token == m_delimiter.column_token_vector[i])
				return total_read;// column 끝나는 토큰 발견
		}
	}

	return total_read;
}

int TokenContentReader::strlen_compare(const void* a, const void* b)
{
	const _FIND_TOKEN* first = (const _FIND_TOKEN*)a;
	const _FIND_TOKEN* second = (const _FIND_TOKEN*)b;
	if (first->len>second->len)
		return -1;
	else if (first->len<second->len)
		return 1;
	else
		return 0;
}

inline void TokenContentReader::CreateFindTokenVector(const std::vector<const char*> &token_vector, const _change_vector& change_vector, _find_token_vector& find_token_vector)
{
	// 큰 순서대로 정렬한다.
	for (int i = 0; i < token_vector.size(); i++)
	{
		_FIND_TOKEN find_token(token_vector[i]);
		if (find_token.len>0)
			find_token_vector.push_back(find_token);
	}

	for (int i = 0; i < change_vector.size(); i++)
	{
		_FIND_TOKEN find_token(change_vector[i].first.c_str(), change_vector[i].second.c_str());
		if (find_token.len>0)
			find_token_vector.push_back(find_token);
	}

	std::qsort(&find_token_vector[0], find_token_vector.size(), sizeof(_FIND_TOKEN), strlen_compare);
}

inline neuro_size_t TokenContentReader::ReadStringUntilToken(const _find_token_vector& find_token_vector, std::string& found_token, std::string* ret_text)
{
	found_token.clear();
	if (!m_text_buffer)
		return 0;

	neuro_size_t read = 0;

	const int token_size = find_token_vector.size();

	if (token_size == 0 || find_token_vector[0].len == 0)
	{// 토큰이 없으면 그냥 마지막까지 계속 읽어 들인다.
		if (ret_text)
			ret_text->append(m_text_read_cur, m_text_read_last);
		read += m_text_read_last - m_text_read_cur;

		m_text_read_cur = m_text_read_last;
		return true;
	}

	const neuro_size_t min_read_size = find_token_vector[0].len;
	const char* start = m_text_read_cur;
	while (m_text_read_cur<m_text_read_last)
	{
		const _FIND_TOKEN* found = FindToken(find_token_vector, m_text_read_cur, m_text_read_last - m_text_read_cur);
		if (found)
		{
			if (ret_text)
				ret_text->append(start, m_text_read_cur);
			read += m_text_read_cur - start;

			m_text_read_cur += found->len;
			start = m_text_read_cur;

			if (found->change == NULL)
			{	// 바꾸는 token이 아니면 찾은 것이다.
				found_token = found->str;
				break;
			}

			// 단순히 바꾸는 것이다.
			if (ret_text)
				ret_text->append(found->change);
			read += found->change_len;
		}
		else// 아무것도 찾은게 없으므로 다음거 검사
		{
			++m_text_read_cur;
		}
	}
	if (ret_text)
		ret_text->append(start, m_text_read_cur);
	read += m_text_read_cur - start;

	return read;
}

inline const TokenContentReader::_FIND_TOKEN* TokenContentReader::FindToken(const _find_token_vector& find_token_vector, const char* str, size_t len)
{
	for (int i_token = 0, n = find_token_vector.size(); i_token < n; i_token++)
	{
		const _FIND_TOKEN& find_token = find_token_vector[i_token];
		if (len >= find_token.len)
		{
			int i = 0;
			for (; i < find_token.len; i++)
			{
				if (str[i] != find_token.str[i])
					break;
			}
			if (i == find_token.len)
				return &find_token;
		}
	}
	return NULL;
}
