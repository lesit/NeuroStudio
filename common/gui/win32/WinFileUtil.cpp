#include "stdafx.h"

#include "WinFileUtil.h"

using namespace np::gui::win32;

bool WinFileUtil::IsDirectory(const wchar_t* strFilePath)
{
	DWORD dwAttr = GetFileAttributes(strFilePath);
	if (dwAttr == INVALID_FILE_ATTRIBUTES)
		return false;

	return (dwAttr & FILE_ATTRIBUTE_DIRECTORY)>0;
}

bool WinFileUtil::GetNormalFileType(const wchar_t* strFilePath, bool& bDirectory)
{
	DWORD dwAttr = GetFileAttributes(strFilePath);
	if (dwAttr == INVALID_FILE_ATTRIBUTES)
		return false;

	if (dwAttr & (FILE_ATTRIBUTE_TEMPORARY | FILE_ATTRIBUTE_SYSTEM | FILE_ATTRIBUTE_HIDDEN))
		return false;

	bDirectory = (dwAttr & FILE_ATTRIBUTE_DIRECTORY)>0;
	return true;
}

void WinFileUtil::GetSubFiles(const wchar_t* dir_path, std_wstring_vector& path_vector)
{
	WIN32_FIND_DATA findFileData;

	HANDLE hFind = FindFirstFile(dir_path, &findFileData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		DEBUG_OUTPUT(L"FindFirstFile failed (%d)", GetLastError());
		return;
	}

	do
	{
		if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue;

		path_vector.push_back(findFileData.cFileName);

	} while (FindNextFile(hFind, &findFileData));

	FindClose(hFind);
}
