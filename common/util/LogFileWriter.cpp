#include "LogFileWriter.h"

#if defined(WIN32) | defined(WIN64)
#include <windows.h>
#endif

#include "StringUtil.h"
#include "np_util.h"

///////////////////////////////////////////////////////////////////////////////////////
// Logging functions
// add ext start

//bool LogFileWriter::m_bCSGlobalLogInitialized=false;

#if !defined(_LOG_PATH)
	#if defined(SE_EXCEPTION_LOG_PATH)
		#define _LOG_PATH SE_EXCEPTION_LOG_PATH
	#else
		#define _LOG_PATH L"./np_log.txt"
	#endif
#endif

#define MAX_LOG_PATH 1024

wchar_t szGlobalLogPath[1024]={0,};
neuro_u64 dwGlobalLogFileAttributeFlags=0;
char szProcessName[MAX_LOG_PATH]={0,};
char szModuleName[MAX_LOG_PATH]={0,};

bool g_bWriteLog=true;

#ifdef _USE_BIG_LOG
// 디버깅 모드에서는 512 mega byte
#define MAX_LOG_SIZE (1024L*1024*512)
#else
// 최대 2 메가 바이트
#define MAX_LOG_SIZE 1024L*1024*2;
#endif

static __int64 nGlobalMaxSize=MAX_LOG_SIZE;

namespace np
{
	namespace util
	{
		class CStaticLogInitialize
		{
		public:
			CStaticLogInitialize()
			{
				wcscpy(szGlobalLogPath, _LOG_PATH);
				dwGlobalLogFileAttributeFlags = FILE_ATTRIBUTE_NORMAL;
				memset(szProcessName, 0, sizeof(szProcessName));
				memset(szModuleName, 0, sizeof(szModuleName));

				g_bWriteLog = true;
			}
			virtual ~CStaticLogInitialize()
			{
			}

		public:
			void SetProcessName()
			{
				GetModuleFileNameA(NULL, szProcessName, MAX_LOG_PATH);
			}

			void SetModuleName()
			{
				char szModuleFileName[MAX_LOG_PATH];
				GetModuleFileNameA(NULL, szModuleFileName, MAX_LOG_PATH);
				_splitpath(szModuleFileName, NULL, NULL, szModuleName, NULL);
			}
		};
	}
}

CStaticLogInitialize globalLogManager;

LogFileWriter::LogFileWriter(const wchar_t* strLogPath, bool bHidden)
{
	if(!strLogPath)
		m_strLogPath=_LOG_PATH;
	else
		m_strLogPath=strLogPath;

	m_nMaxSize=MAX_LOG_SIZE;

	m_dwFileAttributeFlags = bHidden ? FILE_ATTRIBUTE_HIDDEN : FILE_ATTRIBUTE_NORMAL;

//	InitializeCriticalSection(&m_csLog);
}

LogFileWriter::~LogFileWriter()
{
//	DeleteCriticalSection(&m_csLog);
}

void LogFileWriter::DisableWriteLog()
{
	g_bWriteLog=false;
}

std::wstring LogFileWriter::GetLogPath()
{
	return szGlobalLogPath;
}

void LogFileWriter::SetLogPath(const wchar_t* strPath, bool bWriteProcessName)
{
	wcscpy_s(szGlobalLogPath, strPath);

	if(bWriteProcessName)
		globalLogManager.SetProcessName();
	else
		memset(szProcessName, 0, sizeof(szProcessName));
}

void LogFileWriter::_SetLogPath(const wchar_t* strPath)
{
	m_strLogPath=strPath;
}

void LogFileWriter::Write(const wchar_t* strFormat, ...)
{
	if(!g_bWriteLog)
		return;

	std::wstring text;
	va_list args;
	va_start(args, strFormat);
	text=StringUtil::FormatV(strFormat, args);
	va_end(args);

	WriteString(text.c_str());
}

void LogFileWriter::DebugWrite(const wchar_t* strFormat, ...)
{
#if !defined(_DEBUG)
	return;
#endif

	std::wstring text;
	va_list args;
	va_start(args, strFormat);
	text = StringUtil::FormatV(strFormat, args);
	va_end(args);

	WriteString(text.c_str());
}

void LogFileWriter::Write(LogFileWriter* additionalLogManager, const wchar_t* strFormat, ...)
{
	if(!g_bWriteLog)
		return;

	std::wstring text;
	va_list args;
	va_start(args, strFormat);
	text = StringUtil::FormatV(strFormat, args);
	va_end(args);

	WriteString(text.c_str(), additionalLogManager);
}

void LogFileWriter::WriteString(const wchar_t* text, LogFileWriter* additionalLogManager)
{
	if(!g_bWriteLog)
		return;

	WriteLog(szGlobalLogPath, nGlobalMaxSize, dwGlobalLogFileAttributeFlags, text);

	if(additionalLogManager)
		additionalLogManager->_WriteString(text);
}

void LogFileWriter::_Write(const wchar_t* strFormat, ...)
{
	std::wstring text;
	va_list args;
	va_start(args, strFormat);
	text = StringUtil::FormatV(strFormat, args);
	va_end(args);

	_WriteString(text.c_str());
}

void LogFileWriter::_WriteWithTime(const wchar_t* strFormat, ...)
{
	std::wstring text = L"[";
	text.append(NP_Util::TimeOutput()).append(L"] ");

	va_list args;
	va_start(args, strFormat);
	text.append(StringUtil::FormatV(strFormat, args));
	va_end(args);

	_WriteString(text.c_str());
}

void LogFileWriter::_WriteString(const wchar_t* text)
{
	WriteLog(m_strLogPath.c_str(), m_nMaxSize, m_dwFileAttributeFlags, text);
}

void LogFileWriter::WriteLog(const wchar_t* strLogPath, __int64 nMaxSize, neuro_u64 dwFileAttributeFlag, const wchar_t* text)
{
	BackupLogFile(strLogPath, nMaxSize);
	WriteLogFile(strLogPath, dwFileAttributeFlag, text);
}

void LogFileWriter::BackupLogFile(const wchar_t* strLogPath, __int64 nMaxSize)
{
	if(!strLogPath || wcslen(strLogPath)<=0)
		return;

	HANDLE hFile = CreateFile(strLogPath, 0, 0, NULL, OPEN_EXISTING, 0, NULL);
	if(hFile==INVALID_HANDLE_VALUE)
		return;

	LARGE_INTEGER fileSize;
	BOOL bRet=GetFileSizeEx (hFile, &fileSize);
	CloseHandle(hFile);
	if(!bRet)
		return;

	if(fileSize.QuadPart>=nMaxSize)	// 만약 로그의 파일 크기가 최대 크기를 넘긴다면, 로그를 백업하고 다시 쓰도록 한다.
	{
		std::wstring backup_path = std::wstring(strLogPath) + L".bak";
		CopyFile(strLogPath, backup_path.c_str(), FALSE);
		DeleteFile(strLogPath);
	}
}

void LogFileWriter::WriteLogFile(const wchar_t* strLogPath, neuro_u64 dwFileAttributeFlag, const wchar_t* text)
{
	try
	{
		HANDLE fh=CreateFile(strLogPath, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, dwFileAttributeFlag,NULL);
		if(fh!=INVALID_HANDLE_VALUE)
		{
			DWORD byteswritten;	// don't care really.

			SetFilePointer(fh,0,NULL,FILE_END);

			std::string log = util::StringUtil::WideToMultiByte(text);
			WriteFile(fh, log.c_str(), log.size(), &byteswritten, NULL);
			CloseHandle(fh);
		}
		else
		{
#ifdef _DEBUG
			neuro_u64 dwErr=GetLastError();
			//TRACE(_T("log[%s] error=%d\r\n"), m_strLogPath, dwErr);

			//if(m_strLogPath.Mid(3).CompareNoCase(_T("WaterwallSecurityHistory.wwsd"))==0)
			//	int a=0;
#endif
		}
	}
	catch(...)
	{
//		std::wstring extLog=StringUtil::Format<wchar_t>(_T("Exception occured on remaining log : log[%s]"), log.c_str());
//		WritePrivateProfileString(_T("NORMAL ERROR"), strTime, extLog.c_str(), _T("c:\\np_extenderror.ini"));
	}

//	LeaveCriticalSection(&m_csLog);
}
