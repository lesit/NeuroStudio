#include "FileDeviceAdaptor.h"

#include "util/FileUtil.h"
#include "util/np_util.h"

#if defined(WIN32) | defined(WIN64)
#include <windows.h>
#endif

using namespace np;
using namespace np::device;

FileDeviceAdaptor::FileDeviceAdaptor()
{
	m_hFile=INVALID_HANDLE_VALUE;
}
			
FileDeviceAdaptor::~FileDeviceAdaptor(void)
{
	Close();
}

std::string FileDeviceAdaptor::GetDeviceName() const
{
	return util::FileUtil::GetFileName(m_file_path.c_str());
}

bool FileDeviceAdaptor::Create(const char* strFilePath, bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size)
{
	if (!strFilePath)
		return false;

	if (!strFilePath || strlen(strFilePath) == 0)
		return false;

	if (m_hFile != INVALID_HANDLE_VALUE)
	{
		Close();
		if (m_hFile != INVALID_HANDLE_VALUE)
			return false;
	}

	neuro_u32 access = GENERIC_READ;
	neuro_u32 share = FILE_SHARE_READ;
	neuro_u32 create_diposition = OPEN_EXISTING;
	if (!bReadOnly)
	{
		access |= GENERIC_WRITE;
		create_diposition = bCreateAlways ? CREATE_ALWAYS : OPEN_ALWAYS;
		if (bShareWrite)
			share |= FILE_SHARE_WRITE;
	}

	m_hFile = CreateFile(util::StringUtil::MultiByteToWide(strFilePath).c_str(),
		access,          // open for writing
		share,              // share
		NULL,                   // default security
		create_diposition,
		FILE_ATTRIBUTE_NORMAL,// | FILE_FLAG_NO_BUFFERING ,  // normal file
		NULL);                  // no attr. template

	if (m_hFile == INVALID_HANDLE_VALUE)
	{
		DEBUG_OUTPUT(L"failed create file[%s]. err=%u", util::StringUtil::MultiByteToWide(strFilePath).c_str(), GetLastError());

		return false;
	}

	m_file_path = strFilePath;
	return true;
}

void FileDeviceAdaptor::Close()
{
	if(m_hFile!=INVALID_HANDLE_VALUE)
	{
		FILETIME _creationTime, _lastAccessTime, _lastWriteTime;
		if(::GetFileTime(m_hFile, &_creationTime, &_lastAccessTime, &_lastWriteTime))
		{
			SYSTEMTIME	st;
			GetSystemTime(&st);

			FILETIME ft;
			SystemTimeToFileTime(&st, &ft);

			SetFileTime(m_hFile, NULL, NULL, &ft);
		}

		CloseHandle(m_hFile);

		m_hFile=INVALID_HANDLE_VALUE;
	}
}

neuro_u64 FileDeviceAdaptor::GetMaxExtensibleSize() const
{
	if(m_hFile==INVALID_HANDLE_VALUE)
		return 0;

	wchar_t szRootPath[4] = {(wchar_t)m_file_path.c_str()[0], L':', L'\\', 0};

	ULARGE_INTEGER freeSpaceSize;
	ULARGE_INTEGER totalNumberOfBytes;
	ULARGE_INTEGER totalFreeSpaceSize;
	if(!GetDiskFreeSpaceEx(szRootPath, &freeSpaceSize, &totalNumberOfBytes, &totalFreeSpaceSize))
		return 0;

	return freeSpaceSize.QuadPart;
}

bool FileDeviceAdaptor::SetPosition(neuro_u64 pos)
{
	LARGE_INTEGER lPos;lPos.QuadPart=pos;
	if(!SetFilePointerEx(m_hFile, lPos, NULL, FILE_BEGIN))
	{
		neuro_u32 dwErr=GetLastError();
		return false;
	}
	return true;
}

neuro_u64 FileDeviceAdaptor::GetPosition() const
{
	LARGE_INTEGER ret;

	LARGE_INTEGER pos;
	pos.QuadPart = 0;

	SetFilePointerEx(m_hFile, pos, &ret, FILE_CURRENT);

	return ret.QuadPart;
}

neuro_u64 FileDeviceAdaptor::GetUsageSize() const
{
	LARGE_INTEGER fileSize;
	if (!GetFileSizeEx(m_hFile, &fileSize))
		return 0;

	return fileSize.QuadPart;
}

neuro_u32 FileDeviceAdaptor::Read(void* buffer, neuro_u32 nSize) const
{
	DWORD nReaded=0;
	if(!ReadFile(m_hFile, buffer, nSize, &nReaded, NULL))
	{
		DEBUG_OUTPUT(L"failed read. error=%u", GetLastError());
		return 0;
	}
	return nReaded;
}

neuro_u32 FileDeviceAdaptor::Write(const void* buffer, neuro_u32 nSize)
{
	DWORD nWritten = 0;
	if(!WriteFile(m_hFile, buffer, nSize, &nWritten, NULL))
	{
		neuro_u32 dwErr=GetLastError();
		return 0;
	}
	return nWritten;
}

bool FileDeviceAdaptor::Flush()
{
	return FlushFileBuffers(m_hFile);
}

FileDeviceFactory::FileDeviceFactory(const char* strFilePath)
: m_file_path(strFilePath)
{
}

FileDeviceFactory::~FileDeviceFactory()
{
}

void FileDeviceFactory::Reset()
{
	DeleteFile(util::StringUtil::MultiByteToWide(m_file_path).c_str());
}

bool FileDeviceFactory::operator == (const IODeviceFactory& right) const
{
	if (right.GetType() != _device_type::file)
		return false;

	return m_file_path.compare(((const FileDeviceFactory&)right).m_file_path) == 0;
}

DeviceAdaptor* FileDeviceFactory::Create(bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size)
{
	FileDeviceAdaptor* device = new FileDeviceAdaptor;
	if (!device)
		return NULL;
	if (device->Create(m_file_path.c_str(), bReadOnly, bCreateAlways, bShareWrite))
		return device;

	delete device;
	return NULL;
}
