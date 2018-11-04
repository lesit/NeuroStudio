#if !defined _STRING_UTIL_H
#define _STRING_UTIL_H

#include "../np_types.h"
#include <stdarg.h>
#include <algorithm>

namespace np
{
	namespace util
	{
		enum class _str_utf8_type{ ansi, utf8, failure };

		static const wchar_t* w_default_word_escapes = L" \r\n\t\v\f";
		static const char* default_word_escapes = " \r\n\t\v\f";
		class StringUtil
		{
		public:
			template<typename type>
			static std::basic_string<type> Format(const type* fmt, ...)
			{
				va_list vl;

				va_start(vl, fmt);
				std::basic_string<type> ret=FormatV(fmt, vl);
				va_end(vl);
				return ret;
			}

			static std::wstring FormatV(const wchar_t* fmt, va_list vl)
			{
				int size = _vsnwprintf(0, 0, fmt, vl) + sizeof(wchar_t);

				wchar_t* buffer = new wchar_t[size];

				size = _vsnwprintf(buffer, size, fmt, vl);

				std::wstring ret(buffer, size);
				delete[] buffer;

				return ret;
			}

			static std::string FormatV(const char* fmt, va_list vl)
			{
				int size = _vsnprintf(0, 0, fmt, vl) + sizeof(char);

				char* buffer = new char[size];

				size = _vsnprintf(buffer, size, fmt, vl);

				std::string ret(buffer, size);
				delete[] buffer;

				return ret;
			}

			template<typename type>
			static std::basic_string<type> Transform(neuro_float value)
			{
				return Transform<type>(value);
			}
			template<typename type>
			static std::basic_string<type> Transform(neuro_32 value)
			{
				return Transform<type>(value);
			}
			template<typename type>
			static std::basic_string<type> Transform(neuro_u32 value)
			{
				return Transform<type>(value);
			}
			template<typename type>
			static std::basic_string<type> Transform(neuro_64 value)
			{
				return Transform<type>(value);
			}
			template<typename type>
			static std::basic_string<type> Transform(neuro_u64 value)
			{
				return Transform<type>(value);
			}

			template<>
			static std::wstring Transform<wchar_t>(neuro_float value)
			{
				return Format<wchar_t>(L"%f", double(value));
			}

			template<>
			static std::wstring Transform<wchar_t>(neuro_32 value)
			{
				return Format<wchar_t>(L"%u", value);
			}

			template<>
			static std::wstring Transform<wchar_t>(neuro_u32 value)
			{
				return Format<wchar_t>(L"%u", value);
			}

			template<>
			static std::wstring Transform<wchar_t>(neuro_64 value)
			{
				return Format<wchar_t>(L"%lld", value);
			}

			template<>
			static std::wstring Transform<wchar_t>(neuro_u64 value)
			{
				return Format<wchar_t>(L"%llu", value);
			}

			template<>
			static std::string Transform<char>(neuro_float value)
			{
				return Format<char>("%f", double(value));
			}

			template<>
			static std::string Transform<char>(neuro_32 value)
			{
				return Format<char>("%u", value);
			}

			template<>
			static std::string Transform<char>(neuro_u32 value)
			{
				return Format<char>("%u", value);
			}

			template<>
			static std::string Transform<char>(neuro_64 value)
			{
				return Format<char>("%lld", value);
			}

			template<>
			static std::string Transform<char>(neuro_u64 value)
			{
				return Format<char>("%llu", value);
			}

			static std::wstring Transform(_data_type type, neuro_float value);

			inline static std::wstring FloatString(neuro_float value, neuro_u32 limit=2)
			{
				std::wstring format = value > 0 ? Format<wchar_t>(L"%%.%uf", limit) : L"%f";
				return Format<wchar_t>(format.c_str(), value);
			}

			static std::string TransformTextToAscii(const std::string& str);
			static std::string TransformAsciiToText(const std::string& str);

//			static std::wstring MakeValueLogString(neuron_value* values, neuro_u64 count, const wchar_t* sep = L" ");

			static size_t CharsetTokenizeString(const std::wstring& str, std_wstring_vector& str_vector, const wchar_t* separators = w_default_word_escapes);
			static size_t CharsetTokenizeString(const std::string& str, std_string_vector& str_vector, const char* separators = default_word_escapes);

			static bool IsLangChar(const char* start, const char* last, bool include_digit = false);

			static bool IsString(const char buffer[], size_t nBufferSize);

			static bool IsNumeric(const char buffer[], size_t nBufferSize);
			static bool IsNumeric(const wchar_t buffer[], size_t nBufferSize);

			template<typename type>
			static std::basic_string<type> ToLower(const std::basic_string<type>& str)
			{
				std::basic_string<type> ret;
				ret.resize(str.size());
				std::transform(str.begin(), str.end(), ret.begin(), ::tolower);
				return ret;
			}

			static std::wstring Trim(const std::wstring& str, const wchar_t* escape = w_default_word_escapes);
			static std::string Trim(const std::string& str, const char* escape = default_word_escapes);

			static void Trim(std::wstring& str, const wchar_t* escape = w_default_word_escapes);
			static void Trim(std::string& str, const char* escape = default_word_escapes);

			static int GetLineIndex(const wchar_t* str, int nLine);

			static int GetLineCount(const wchar_t* str);

			static _str_utf8_type CheckUtf8(unsigned char* text, int size);
			static const unsigned char* FindLastTruncatedMultibytesBegin(const unsigned char* text_last, neuro_u64 size);

#if (defined(_WIN32) | defined(_WIN64))&& !defined(__CYGWIN__)
			static int MultiByteToWide(const char* input, int in_len, wchar_t* output, int out_len, bool is_utf8 = true);
			static wchar_t* MultiByteToWide(const char* input, int in_len, int& out_len, bool is_utf8 = true);
			static bool MultiByteToWide(const char* input, int in_len, std::wstring& ret, bool is_utf8 = true);
			static bool MultiByteToWide(const std::string &input, std::wstring& ret, bool is_utf8 = true);
			static std::wstring MultiByteToWide(const std::string &input, bool is_utf8 = true);

			static bool WideToMultiByte(const wchar_t* input, int in_len, std::string &out, bool is_utf8 = true);
			static char* WideToMultiByte(const wchar_t* input, int in_len, int& out_len, bool is_utf8 = true);
			static bool WideToMultiByte(const std::wstring &input, std::string &out, bool is_utf8 = true);
			static std::string WideToMultiByte(const std::wstring &input, bool is_utf8 = true);
#endif
		};
	}
}
#endif
