#pragma once

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CheckDirectory
			{
			public:
				CheckDirectory();
				bool CheckDirPath(CString strFolderPath);

				static std::string BrowserSelectFolder(CWnd* parent, const char* default_dir, const wchar_t* title);

			private:
				bool MakeSureDirectoryDirect(LPCTSTR strFolderPath);

				bool HasPath(const wchar_t* path) const;
				void SetPath(const wchar_t* path);

				CMapStringToPtr m_directoryMap;
			};
		}
	}
}
