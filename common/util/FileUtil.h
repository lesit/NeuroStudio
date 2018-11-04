#if !defined(_FILE_UTIL_H)
#define _FILE_UTIL_H

#include "StringUtil.h"

namespace np
{
	namespace util
	{
		class FileUtil
		{
		public:
			template<typename type>
			static std::basic_string<type> TransformPath(const std::basic_string<type>& path)
			{
				std::basic_string<type> ret;
				for (size_t i = 0; i<path.size(); i++)
					ret.push_back(TransCommonPathChar<type>(path[i]));

				return ret;
			}

			template<typename type>
			static const type* GetFileName(const type* file_path)
			{
				const type* name = file_path;
				while (*name != '\0') name++;

				while ((--name) >= file_path)
				{
					if (*name == '/' || *name == '\\')
						return name + 1;
				}
				return file_path;
			}

			template<typename type>
			static std::basic_string<type> GetNameFromFileName(std::basic_string<type> file_name)
			{
				size_t last_dot = file_name.rfind('.');
				if (last_dot == std::basic_string<type>::npos)
					return file_name;
				file_name.erase(file_name.begin() + last_dot, file_name.end());
				return file_name;
			}

			template<typename type>
			static std::basic_string<type> GetExtFromFileName(std::basic_string<type> file_name)
			{
				size_t last_dot = file_name.rfind('.');
				if (last_dot == std::basic_string<type>::npos)
					return "";
				
				return std::basic_string<type>(file_name.begin() + last_dot, file_name.end());
			}

			template<typename type>
			static bool ComparePath(const std::basic_string<type>& path1, const std::basic_string<type>& path2)
			{
				std::basic_string<type> s_path1 = TransformPath<type>(path1);
				std::basic_string<type> s_path2 = TransformPath<type>(path2);
				return s_path1 == s_path2;
			}

			template<typename type>
			static std::basic_string<type> GetDirFromPath(const std::basic_string<type>& path)
			{
				const type* start = path.c_str();
				const type* ch = start + path.size();

				while ((--ch) >= start)
				{
					if (*ch == '/' || *ch == '\\')
						return std::basic_string<type>(path.begin(), path.begin() + (ch-start));
				}

				return path;
			}

			template<typename type>
			static std::basic_string<type> GetRelativePath(std::basic_string<type> base_dir, const std::basic_string<type>& path)
			{
				std::basic_string<type> ret = TransformPath<type>(path);
				const type* ptr = ret.c_str();
				const type* base = base_dir.c_str();
				size_t base_len = base_dir.length();
				while (*ptr != 0 && *base != '\0')
				{
					type ch = *(ptr++);
					type base_ch = TransCommonPathChar(*(base++));

					if (ch != base_ch)	// not completed include base_dir. ex) base_dir=c:/temp/abc/	path=c:/temp/efg/bbb.txt
						return path;
				}

				if (*base != 0)	// not completed cacn base_dir. so, path is bigger than base_dir. ex) base_dir=c:/temp/abc/	path=c:/temp/bbb.txt
					return path;

				return ptr;
			}

			template<typename type>
			static std::basic_string<type> GetFilePath(const std::basic_string<type>& base_dir, const std::basic_string<type>& relative_path)
			{
				if (relative_path.size() == 0)
					return std::basic_string<type>();

				if (relative_path[0] == '/')		// unix, linux, mac root
					return relative_path;
				else if (relative_path.size() >= 2 && relative_path[1] == ':')	// windows root
					return relative_path;
				else if (relative_path[0] == '.')
					return relative_path;
				else
					return std::basic_string<type>(base_dir).append(relative_path);
			}

			template<typename type>
			inline static type TransCommonPathChar(type ch)
			{
				return ch;
			}

			template<>
			inline static char TransCommonPathChar<char>(char ch)
			{
				if (ch == '\\')
					ch = '/';
#ifdef _WINDOWS
				else if (ch >= L'A' && ch <= L'Z')
					ch = L'a' + (ch - L'A');
#endif			
				return ch;
			}

			template<>
			inline static wchar_t TransCommonPathChar<wchar_t>(wchar_t ch)
			{
				if (ch == L'\\')
					ch = '/';
#ifdef _WINDOWS
				else if (ch >= L'A' && ch <= L'Z')
					ch = L'a' + (ch - L'A');
#endif			
				return ch;
			}
		};
	}
}

#endif

