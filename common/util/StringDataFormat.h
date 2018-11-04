#if !defined(_STRING_DATA_FORMAT_H)
#define _STRING_DATA_FORMAT_H

#include <time.h>

#include "common.h"

namespace np
{
	namespace dp
	{
		static _data_type DataTypeTest(const wchar_t* str)
		{
			neuro_u32 slash, minus, dot, nchar;
			slash = minus = dot = nchar = 0;

			size_t len = wcslen(str);
			if (len == 0)
				return _data_type::string;

			for (size_t i = 0; i < len; i++)
			{
				wchar_t ch = str[i];
				if (ch == L'/')
					++slash;
				else if (ch == L'-')
					++minus;
				else if (ch == L'.')
					++dot;
				else if (ch == L'%')
					return _data_type::percentage;
				else if (ch != L',' && ch<L'0' || ch>L'9')
					++nchar;
			}

			if (slash == 2 || minus == 2 || dot == 2)
				return _data_type::time;

			if (dot == 1)
				return _data_type::float32;

			if (nchar == 0)
				return _data_type::int64;

			wchar_t temp[100];

			neuro_u64 iret = _wtoi64(str);
			wsprintf(temp, L"%llu", iret);
			/*	모두 정확히 찾을려면 "%I32d"로도 변경해서 비교해 봐야 하지만
				어짜피 모든 데이터에서 찾는 것도 아니기 때문에 비교만 64bit로 하고(32bit도 비교가 되기 때문에) 기본형인 32bit로 해준다.
				*/
			if (wcscmp(temp, str) == 0)
				return _data_type::int64;

			return _data_type::string;
		}

		static neuro_float StringTransform(_data_type type, const char* str)
		{
			if (type == _data_type::time)
			{
				tm t = { 0, };
				sscanf_s(str, "%u-%u-%u", &t.tm_year, &t.tm_mon, &t.tm_mday);
				t.tm_year -= 1900;
				t.tm_mon -= 1;
				return static_cast<neuro_float>(_mktime64(&t));
			}
			else if (type == _data_type::float32 || type == _data_type::float64)
				return static_cast<neuro_float>(atof(str));
			else if (type == _data_type::percentage)
				return static_cast<neuro_float>(atof(str) / 100.0);
			else if (type == _data_type::int32 || type == _data_type::int64)
			{
				__int64 value = 0;
				char ch;
				while ((ch = *str++) != NULL)
				{
					if (ch == ',')
						continue;
					if (!isdigit(ch))
					{
						value = 0;
						break;
					}
					value = value * 10 + (ch - '0');
				}
				return static_cast<neuro_float>(value);
			}
			else
				return 0.0;
		}
	}
}

#endif
