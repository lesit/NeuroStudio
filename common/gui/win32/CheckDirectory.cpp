#include "stdafx.h"

#include "CheckDirectory.h"
#include "WinFileUtil.h"

using namespace np::gui::win32;

CheckDirectory::CheckDirectory()
{
}

bool CheckDirectory::CheckDirPath(CString strFolderPath)
{
	if(strFolderPath.IsEmpty())
		return false;

	strFolderPath.Replace(_T('/'), _T('\\'));

	if(strFolderPath[strFolderPath.GetLength()-1] != _T('\\'))
		strFolderPath+=_T('\\');
		
	if(HasPath(strFolderPath))
		return true;

	typedef BOOL (WINAPI *MAKESUREDIRECTORYPATHEXISTS)(IN PCSTR strDirPath);

	HMODULE	hDllHandle = LoadLibrary(_T("DBGHELP.DLL"));
	if(hDllHandle)
	{
		MAKESUREDIRECTORYPATHEXISTS MakeSureDirectoryPathExists = (MAKESUREDIRECTORYPATHEXISTS) GetProcAddress(hDllHandle, "MakeSureDirectoryPathExists");

		if(MakeSureDirectoryPathExists)
		{
			bool bRet=false;
			if(MakeSureDirectoryPathExists(CStringA(strFolderPath)))
			{
				SetPath(strFolderPath);
				bRet=true;
			}
			else
			{
				DWORD nErr=GetLastError();
				TRACE(_T("CFileUtil::CheckDirectory : failed to MakeSureDirectoryPathExists(%s) : error[%d]"), strFolderPath, nErr);
			}

			FreeLibrary(hDllHandle);
			return bRet;
		}
		FreeLibrary(hDllHandle);
	}

	return MakeSureDirectoryDirect(strFolderPath);
}

bool CheckDirectory::MakeSureDirectoryDirect(LPCTSTR strFolderPath)
{
	if(HasPath(strFolderPath))
		return true;

	bool bDirectory;
	if(WinFileUtil::GetNormalFileType(strFolderPath, bDirectory))
	{
		if(bDirectory)
		{
			SetPath(strFolderPath);
			return true;
		}

		// path가 있는데 파일인 경우 실패
		return false;
	}

	{
		LPCTSTR strLast=strFolderPath+_tcslen(strFolderPath)-1;
		if(*strLast==_T('\\') || *strLast==_T('/'))	// 맨 마지막의 '/'는 지나간다.
			--strLast;

		for(;strLast>strFolderPath;strLast--)
		{
			if(*strLast==_T('\\') || *strLast==_T('/'))
			{
				CString strUpperPath(strFolderPath, strLast-strFolderPath);
				if(!CheckDirPath(strUpperPath))
					return false;

				break;
			}
		}
	}

	if(!::CreateDirectory(strFolderPath, NULL))
	{
		DWORD nErr=GetLastError();
		if(nErr==183)	// 이미 존재한다.
		{
			SetPath(strFolderPath);
			return true;
		}

		TRACE(_T("CFileUtil::CheckDirectory : failed to CreateDirectory(%s) : error[%d]"), strFolderPath, nErr);
		return false;
	}

	if(!WinFileUtil::GetNormalFileType(strFolderPath, bDirectory) || !bDirectory)
	{
		TRACE(_T("CFileUtil::CheckDirectory : %s is not directory"), strFolderPath);
		return false;
	}

	SetPath(strFolderPath);
	return true;
}

bool CheckDirectory::HasPath(const wchar_t* path) const
{
	CString nocase(path);
	nocase.MakeUpper();
	
	return m_directoryMap.HashKey(nocase) != FALSE;
}

void CheckDirectory::SetPath(const wchar_t* path)
{
	CString nocase(path);
	nocase.MakeUpper();

	m_directoryMap.SetAt(nocase, NULL);
}

int CALLBACK CallBackProcBrowser(HWND hWnd, UINT Msg, LPARAM unused, LPARAM lpData)
{
	if (Msg == BFFM_INITIALIZED) {
		::SendMessage(hWnd, BFFM_SETSELECTION, TRUE, lpData);
	}
	return 0;
}

std::string CheckDirectory::BrowserSelectFolder(CWnd* parent, const char* default_dir, const wchar_t* title)
{
	BROWSEINFO bi;
	::ZeroMemory(&bi, sizeof(BROWSEINFO));

	if (parent)
		bi.hwndOwner = parent->GetSafeHwnd();

	bi.lpszTitle = title;
	bi.ulFlags = BIF_NEWDIALOGSTYLE | BIF_RETURNONLYFSDIRS | BIF_DONTGOBELOWDOMAIN;

	wchar_t buffer[MAX_PATH];
	bi.pszDisplayName = buffer;

	CString Path(util::StringUtil::MultiByteToWide(default_dir).c_str());

	bi.lParam = (LPARAM)(LPCTSTR)Path; // set lParam to point to path
	bi.lpfn = CallBackProcBrowser;	// set the callback procedure

	ITEMIDLIST *idl = SHBrowseForFolder(&bi);
	if (idl) {
		SHGetPathFromIDList(idl, buffer);	// get path string from ITEMIDLIST

		std::string ret(util::StringUtil::WideToMultiByte(buffer));

		LPMALLOC lpm;
		if (SHGetMalloc(&lpm) == NOERROR)
			lpm->Free(idl);    // free memory returned by SHBrowseForFolder

		return ret;
	}
	return "";
}
