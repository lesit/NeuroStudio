#pragma once

#include "../np_types.h"

namespace np
{
	namespace util
	{
		class LogFileWriter
		{
		public:
			LogFileWriter(const wchar_t* strLogPath = NULL, bool bHidden = false);
			virtual ~LogFileWriter();

			void _WriteString(const wchar_t* text);
			void _Write(const wchar_t* strFormat, ...);
			void _WriteWithTime(const wchar_t* strFormat, ...);

			std::wstring _GetLogPath(){ return m_strLogPath; }
			void _SetLogPath(const wchar_t* strPath);

			void _SetLogSize(__int64 nMaxSize){ m_nMaxSize = nMaxSize; }
		public:
			static void DisableWriteLog();
			static std::wstring GetLogPath();
			static void SetLogPath(const wchar_t* strPath, bool bWriteProcessName = true);

			static void Write(const wchar_t* strFormat, ...);
			static void Write(LogFileWriter* additionalLogManager, const wchar_t* strFormat, ...);
			static void DebugWrite(const wchar_t* strFormat, ...);

			static void WriteString(const wchar_t* text, LogFileWriter* additionalLogManager = NULL);

		private:
			static void WriteLog(const wchar_t* strLogPath, __int64 nMaxSize, neuro_u64 dwFileAttributeFlag, const wchar_t* text);
			static void BackupLogFile(const wchar_t* strLogPath, __int64 nMaxSize);
			static void WriteLogFile(const wchar_t* strLogPath, neuro_u64 dwFileAttributeFlag, const wchar_t* text);

		private:
			std::wstring m_strLogPath;

			__int64 m_nMaxSize;
			neuro_u64 m_dwFileAttributeFlags;
		};
	};
}
using namespace np::util;

#if !defined(FileLogWrite)
#define LogFileWrite(fmt, ...)	LogFileWriter::Write(std::wstring(_T("%s : ")) + fmt, std::wstring(__FUNCTION__), __VA_ARGS__);
#endif
