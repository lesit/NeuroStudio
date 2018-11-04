#pragma once

#include "common.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class WinFileUtil
			{
			public:
				static bool IsDirectory(const wchar_t* strFilePath);
				static bool GetNormalFileType(const wchar_t* strFilePath, bool& bDirectory);

				static void GetSubFiles(const wchar_t* dir_path, std_wstring_vector& path_vector);
			};
		}
	}
}