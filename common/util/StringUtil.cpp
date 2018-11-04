#include "StringUtil.h"

#include "np_util.h"

#if (defined(_WIN32) | defined(_WIN64))&& !defined(__CYGWIN__)
#include <Windows.h>
#endif

using namespace np::util;

std::string StringUtil::TransformTextToAscii(const std::string& str)
{
	std::string ret;

	std::string::size_type start = 0;
	std::string::size_type end = str.size();
	std::string::size_type pos = 0;
	while (start<end && (pos = str.find_first_of("\\", start)) != std::string::npos)
	{
		ret.append(str.begin() + start, str.begin() + pos);

		start = pos + 1;
		if (start >= end)
			break;

		char ch = str[start];
		switch (ch)
		{
		case L'r':
			ch = '\r';
			break;
		case L'n':
			ch = '\n';
			break;
		case L't':
			ch = '\t';
			break;
		case L'\"':
			ch = '\"';
			break;
		default:
			ch = ' ';
		}
		ret += ch;
		++start;
	}
	ret.append(str.begin() + start, str.begin() + end);
	return ret;
}

std::string StringUtil::TransformAsciiToText(const std::string& str)
{
	std::string ret;

	std::wstring::size_type start = 0;
	std::wstring::size_type end = str.size();
	std::wstring::size_type pos = 0;
	while (start<end&& (pos = str.find_first_of("\r\n\t\"", start)) != std::wstring::npos)
	{
		ret.append(str.begin() + start, str.begin() + pos);
		ret.append("\\");
		if (str[pos] == '\r')
			ret.append("r");
		else if (str[pos] == '\n')
			ret.append("n");
		else if (str[pos] == '\t')
			ret.append("t");
		else if (str[pos] == '\"')
			ret.append("\"");

		start = pos + 1;
	}
	ret.append(str.begin() + start, str.begin() + end);
	return ret;
}

std::wstring StringUtil::Transform(_data_type type, neuro_float value)
{
	switch(type)
	{
	case _data_type::int32:
		return Transform<wchar_t>((neuro_u32)value);
	case _data_type::int64:
		return Transform<wchar_t>((neuro_64)value);
	case _data_type::float32:
		return Transform<wchar_t>((neuro_float)value);
	case _data_type::percentage:
		return Format<wchar_t>(L"%f %%", (float)value);
	}
	return L"";
}
/*
std::wstring StringUtil::MakeValueLogString(neuron_value* values, neuro_u64 count, const wchar_t* sep)
{
	std::wstring str;
	for (neuro_u64 i = 0; i<count; i++)
	{
		str += util::StringUtil::Transform<wchar_t>(values[i]);
		str += L" ";
	}
	return str;
}
*/

namespace np
{
	namespace util
	{
		template<typename char_type, class string_type>
		inline size_t TokenizeString(const string_type& str, std::vector<string_type>& str_vector, const char_type* separators)
		{
			str_vector.clear();

			size_t start = 0;
			while (start<str.size())
			{
				size_t end = str.find_first_of(separators, start);
				if (end == std::string::npos)
					end = str.size();

				str_vector.push_back(string_type(str.begin() + start, str.begin() + end));
				start = end + 1;
			}
			return str_vector.size();
		}
	}
}

size_t StringUtil::CharsetTokenizeString(const std::wstring& str, std_wstring_vector& str_vector, const wchar_t* separators)
{
	return TokenizeString<wchar_t>(str, str_vector, separators);
}

size_t StringUtil::CharsetTokenizeString(const std::string& str, std_string_vector& str_vector, const char* separators)
{
	return TokenizeString<char>(str, str_vector, separators);
}

bool StringUtil::IsLangChar(const char* start, const char* last, bool include_digit)
{
	const char* p = start;
	for (; *p != '\0' && (last==NULL || p!=last); p++)
	{
		if (include_digit && isdigit(*p))
			continue;

		if (isalpha(*p))
			continue;
		
		if ((*p & 0x80) != 0)
			continue;
		
		return false;
	}
	return p>start;
}

bool StringUtil::IsString(const char buffer[], size_t nBufferSize)
{
	size_t i = 0;
	for(;i<nBufferSize;i++)
	{
		char ch=buffer[i];
		if (ch == 0)
			break;

		if (ch >= 32 && ch <= 126)
			continue;

		if(ch & 0x80)	// 한글 등 multibyte의 다국어 문자
			continue;

		return false;
	}
	return i>0;
}

bool StringUtil::IsNumeric(const char buffer[], size_t nBufferSize)
{
	size_t i = 0;
	for (; i<nBufferSize; i++)
	{
		char ch = buffer[i];
		if (ch == 0)
			break;

		if (ch<'0' || ch>'9')
			return false;
	}
	return i>0;
}

bool StringUtil::IsNumeric(const wchar_t buffer[], size_t nBufferSize)
{
	for (size_t i = 0; i < nBufferSize; i++)
	{
		wchar_t ch = buffer[i];
		if (ch == 0)
			break;

		if (ch<L'0' || ch>L'9')
			return false;
	}
	return nBufferSize>0;
}

std::wstring StringUtil::Trim(const std::wstring& str, const wchar_t* escape)
{
	std::wstring ret = str;
	Trim(ret, escape);
	return ret;
}

std::string StringUtil::Trim(const std::string& str, const char* escape)
{
	std::string ret = str;
	Trim(ret, escape);
	return ret;
}

inline void StringUtil::Trim(std::wstring& str, const wchar_t* escape)
{
	if (str.size() == 0)
		return;

	size_t start = str.find_first_not_of(escape);
	if (start == std::wstring::npos)
	{
		str.clear();
		return;
	}
	if (start>0)
		str.erase(str.begin(), str.begin() + start);

	size_t end = str.find_last_not_of(escape);
	if (end == std::wstring::npos)
	{
		// 어라? 앞에서 분명히escape가 아닌게 있었는데.. ㅡㅡ
		str.clear();
		return;
	}
	str.erase(str.begin() + end + 1, str.end());
}

inline void StringUtil::Trim(std::string& str, const char* escape)
{
	if (str.size() == 0)
		return;

	size_t start = str.find_first_not_of(escape);
	if (start == std::string::npos)
	{
		str.clear();
		return;
	}
	if(start>0)
		str.erase(str.begin(), str.begin() + start);

	size_t end = str.find_last_not_of(escape);
	if (end == std::string::npos)
	{
		// 어라? 앞에서 분명히escape가 아닌게 있었는데.. ㅡㅡ
		str.clear();
		return;
	}
	str.erase(str.begin() + end + 1, str.end());
}

int StringUtil::GetLineIndex(const wchar_t* str, int nLine)
{
	const wchar_t* find=str;
	for(int i=0;i<nLine;i++)
	{
		const wchar_t* next=wcschr(find, L'\n');
		if(next==NULL)
			return -1;

		find=next+1;
	}
	return (int)(find-str);
}

int StringUtil::GetLineCount(const wchar_t* str)
{
	if(str==NULL || str[0]==0)
		return 0;

	int nLine=1;
	while(true)
	{
		const wchar_t* next=wcschr(str, L'\n');
		if(next==NULL)
			break;

		str=next+1;

		++nLine;
	}
	return nLine;
}

_str_utf8_type StringUtil::CheckUtf8(unsigned char* text, int size)
{
	unsigned char* p = text;
	unsigned char* last = p + size;

	bool has80 = false;
	for (; p < last; p++)
	{
		if ((*p & 0x80) != 0x80)
		{
			// 최상위 비트가 1이 아니면 ASCII 이다.
			continue;
		}

		// 상위 비트가 110이고 다음 문자의 상위 비트가 10이면 UTF8맞음
		// p가 문서 끝을 넘거나 중간에 하나라도 규칙에 맞지 않으면 UTF8이 아님
		if ((*p & 0xe0) == 0xc0)
		{
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
		}
		else if ((*p & 0xf0) == 0xe0)
		{	// 상위 비트가 1110일 때는 다음 두 문자의 상위 비트가 10이어야 한다.
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
		}
		else if ((*p & 0xf8) == 0xf0)
		{	// 상위 5비트가 11110일 때는 다음 세 문자의 상위 비트가 10이어야 한다.
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
			if (++p >= last)
				break;
			if ((*p & 0xc0) != 0x80)
				return _str_utf8_type::failure;
		}
		else
		{
			// 0x80을 넘었는데 상위 비트가 110, 1110, 11110 중 하나가 아니면
			// UTF-8 문서가 아니다.
			DEBUG_OUTPUT(L"failue check utf8");
			return _str_utf8_type::failure;
		}
		has80 = true;
	}

	// 0x80 넘는 값이 하나도 없으면 ANSI로 취급한다.
	if (has80 == false)
		return _str_utf8_type::ansi;

	// 0x80을 넘는 모든 값이 UTF-8의 조건을 만족하면 UTF-8문서이다.
	return _str_utf8_type::utf8;
}

const unsigned char* StringUtil::FindLastTruncatedMultibytesBegin(const unsigned char* first, neuro_u64 size)
{
	if (first == NULL)
		return NULL;

	const unsigned char* last = first + size;

	const unsigned char* p = last;
	while (--p > first)
	{
		if ((*p & 0xc0) != 0x80)
		{
			// 잘린 multibyte의 시작을 찾았다.
			neuro_size_t last_index = first + size - p;

			// 만약 다 완성된 문자라면 잘린게 아니다!
			if ((*p & 0xe0) == 0xc0 && last_index == 2)
				return last;
			if ((*p & 0xf0) == 0xe0 && last_index == 3)
				return last;
			if ((*p & 0xf8) == 0xf0 && last_index == 4)
				return last;

			return p;
		}
	}
	return NULL;
}

#if (defined(_WIN32) | defined(_WIN64))&& !defined(__CYGWIN__)
int StringUtil::MultiByteToWide(const char* input, int in_len, wchar_t* output, int out_len, bool is_utf8)
{
	return ::MultiByteToWideChar(is_utf8 ? CP_UTF8 : CP_ACP, 0, input, in_len,
			output, out_len);
}

wchar_t* StringUtil::MultiByteToWide(const char* input, int in_len, int& out_len, bool is_utf8)
{
	out_len = MultiByteToWideChar(is_utf8 ? CP_UTF8 : CP_ACP, 0,
		input, in_len, NULL, 0);
	if (out_len == 0)
		return NULL;
	if (in_len < 0)
		--out_len;
	else
		int a = 0;

	wchar_t* output_encoded = new wchar_t[out_len+1];
	out_len = ::MultiByteToWideChar(is_utf8 ? CP_UTF8 : CP_ACP, 0, input, in_len,
		output_encoded, out_len+1);
	if (out_len == 0)
	{
		delete[] output_encoded;
		return NULL;
	}

	if (in_len < 0)
		--out_len;
	output_encoded[out_len] = L'\0';
	return output_encoded;
}

bool StringUtil::MultiByteToWide(const char* input, int in_len, std::wstring& ret, bool is_utf8)
{
	int out_len;
	wchar_t* output = MultiByteToWide(input, in_len, out_len, is_utf8);
	if (output == NULL)
		return false;

	ret.assign(output, output + out_len);
	delete[] output;

	return true;
}

bool StringUtil::MultiByteToWide(const std::string &input, std::wstring& ret, bool is_utf8)
{
	return MultiByteToWide(input.c_str(), input.size(), ret, is_utf8);
}

std::wstring StringUtil::MultiByteToWide(const std::string &input, bool is_utf8)
{
	std::wstring output;
	MultiByteToWide(input, output, is_utf8);
	return output;
}

char* StringUtil::WideToMultiByte(const wchar_t* input, int in_len, int& out_len, bool is_utf8)
{
	out_len = ::WideCharToMultiByte(is_utf8 ? CP_UTF8 : CP_ACP, 0,
		input, in_len, NULL, 0,
		NULL, NULL);
	if (out_len == 0)
		return false;
	if (in_len < 0)
		--out_len;

	char* output_encoded = new char[out_len + 1];
	out_len = ::WideCharToMultiByte(is_utf8 ? CP_UTF8 : CP_ACP, 0, input, in_len,
		output_encoded,
		out_len+1, NULL, NULL);

	if (out_len == 0)
	{
		delete[] output_encoded;
		return NULL;
	}

	if (in_len < 0)
		--out_len;
	output_encoded[out_len] = '\0';

	return output_encoded;
}

bool StringUtil::WideToMultiByte(const wchar_t* input, int in_len, std::string &out, bool is_utf8)
{
	int out_len;
	char* input_encoded = WideToMultiByte(input, in_len, out_len, is_utf8);

	if (input_encoded == NULL)
		return false;

	out.assign(input_encoded, input_encoded + out_len);
	delete[] input_encoded;
	return true;
}

bool StringUtil::WideToMultiByte(const std::wstring &input, std::string &out, bool is_utf8)
{
	return WideToMultiByte(input.c_str(), input.size(), out, is_utf8);
}

std::string StringUtil::WideToMultiByte(const std::wstring &input, bool is_utf8)
{
	std::string output;
	WideToMultiByte(input, output, is_utf8);
	return output;
}
#endif
