
#include "np_util.h"
#include <stdarg.h>

#if defined(WIN32) | defined(WIN64)
#include <windows.h>
#endif

#include "StringUtil.h"

using namespace np;

std::wstring NP_Util::TimeOutput()
{
	wchar_t str[100];
	SYSTEMTIME sysTime;
	GetLocalTime(&sysTime);
	swprintf_s(str, L"%04d/%02d/%02d %02d:%02d:%02d: %03d", sysTime.wYear,
		sysTime.wMonth,
		sysTime.wDay,
		sysTime.wHour,
		sysTime.wMinute,
		sysTime.wSecond,
		sysTime.wMilliseconds
		);

	return str;
}

void NP_Util::DebugOutput(const wchar_t* strFormat, ...)
{
	va_list args;
	va_start(args, strFormat);
	VDebugOutput(NULL, NULL, strFormat, args);
	va_end(args);
}

void NP_Util::DebugOutputValues(const neuron_value* buffer, neuro_size_t count, int line_size)
{
	const neuron_value* last_buffer = buffer + count;
	int line = 0;

	for (; buffer < last_buffer; buffer++)	// padding
	{
		DebugOutput(L"%f, ", *buffer);
		if (++line % line_size == 0)
		{
			DebugOutput(L"\r\n");
			line = 0;
		}
	}
}

void NP_Util::DebugOutputValues(const neuro_u32* buffer, neuro_size_t count, int line_size)
{
	const neuro_u32* last_buffer = buffer + count;
	int line = 0;

	for (; buffer < last_buffer; buffer++)	// padding
	{
		DebugOutput(L"%d, ", *buffer);
		if (++line % line_size == 0)
		{
			DebugOutput(L"\r\n");
			line = 0;
		}
	}
}
void NP_Util::DebugOutputWithFunctionName(const char* func_name, const wchar_t* strFormat, ...)
{
	std::string a_func_name(func_name);
	std::wstring strFuncName; strFuncName.assign(a_func_name.begin(), a_func_name.end());

	std::wstring content = strFormat;
	content.append(L"\r\n");

	va_list args;
	va_start(args, strFormat);
	VDebugOutput(&TimeOutput(), strFuncName.c_str(), content.c_str(), args);
	va_end(args);
}

#include "LogFileWriter.h"

bool g_debuglog_filewrite = false;
void NP_Util::SetDebugLogWriteFile(const wchar_t* path)
{
	g_debuglog_filewrite = true;
	util::LogFileWriter::SetLogPath(path, false);
}

void NP_Util::VDebugOutput(const std::wstring* time, const wchar_t* func_name, const wchar_t* strFormat, va_list args)
{
#ifndef _DEBUG
	if (!g_debuglog_filewrite)
		return;
#endif

	std::wstring content;

	if (time && time->size() > 0)
		content.append(L"[").append(*time).append(L"] ");

	if (func_name)
	{
		std::wstring str_func = func_name;
#ifndef _DEBUG
		size_t colon = str_func.find(L"::");
		if (colon != std::wstring::npos)
			str_func.erase(str_func.begin(), str_func.begin()+colon+2);
#endif
		content += str_func;
		content += L" : ";
	}
	content += util::StringUtil::FormatV(strFormat, args);

#if defined(WIN32)|defined(WIN64)
	OutputDebugString(content.c_str());
#endif

	if (g_debuglog_filewrite)
	{
		util::LogFileWriter::Write(content.c_str());
	}
}

neuro_u64 NP_Util::GetAvailableMemory()
{
	const neuro_u64 conv_mb_div = 1024 * 1024;

	MEMORYSTATUSEX statex;
	statex.dwLength = sizeof (statex);

	GlobalMemoryStatusEx(&statex);

	return statex.ullAvailPhys >> 20;
	return statex.ullAvailVirtual >> 20;
}

std::string NP_Util::GetSizeString(neuro_size_t size)
{
	if (size > 1024I64 * 1024 * 1024 * 1024 * 1024 * 99)
		return util::StringUtil::Format("%I64d PB", size / 1024 / 1024 / 1024 / 1024 / 1024);
	else if (size > 1024I64 * 1024 * 1024 * 1024 * 1024)
		return util::StringUtil::Format("%.1f PB", (double)(size / 1024.0 / 1024 / 1024 / 1024 / 1024));
	else if (size > 1024I64 * 1024 * 1024 * 1024 * 99)
		return util::StringUtil::Format("%I64d TB", size / 1024 / 1024 / 1024 / 1024);
	else if (size > 1024I64 * 1024 * 1024 * 1024)
		return util::StringUtil::Format("%.1f TB", (double)(size / 1024.0 / 1024 / 1024 / 1024));
	else if (size > 1024I64 * 1024 * 1024 * 99)
		return util::StringUtil::Format("%I64d GB", size / 1024 / 1024 / 1024);
	else if (size > 1024I64 * 1024 * 1024)
		return util::StringUtil::Format("%.1f GB", (double)(size / 1024.0 / 1024 / 1024));
	else if (size > 1024I64 * 1024 * 99)
		return util::StringUtil::Format("%I64d MB", size / 1024 / 1024);
	else if (size > 1024I64 * 1024)
		return util::StringUtil::Format("%.1f MB", (double)(size / 1024.0 / 1024));
	else if (size >= 1024I64)
		return util::StringUtil::Format("%I64d KB", size / 1024);
	else
		return util::StringUtil::Format("%I64d BYTE", size);
}
