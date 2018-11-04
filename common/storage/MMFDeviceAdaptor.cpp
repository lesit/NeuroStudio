#include "MMFDeviceAdaptor.h"
#include "util/StringUtil.h"

#if defined(WIN32) | defined(WIN64)
#include <windows.h>
#endif

using namespace np::device;
using namespace np;

MMFDeviceAdaptor::MMFDeviceAdaptor()
: m_nMaxBlockSize(2*BYTES_PER_GB)	// 일단 최대 2gb 로 한다.
{
	m_bReadOnly = true;

	m_hFileMapping=NULL;

    SYSTEM_INFO si;
    GetSystemInfo(&si);
	m_dwAllocationGranularity=si.dwAllocationGranularity;

	m_nFileSize=0;

	m_pbFile=NULL;
	m_nMappedPointer=0;
	m_nMappedSize=0;

	m_position = 0;
}

MMFDeviceAdaptor::~MMFDeviceAdaptor()
{
	if(m_pbFile)
		UnmapViewOfFile(m_pbFile);

	if(m_hFileMapping)
		CloseHandle(m_hFileMapping);
}

bool MMFDeviceAdaptor::Create(const char* strFilePath, bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size)
{
	m_position = 0;

	if (!strFilePath || strlen(strFilePath)==0)
		return false;

	if(m_hFile!=INVALID_HANDLE_VALUE)
		return false;

	m_bReadOnly = bReadOnly;

	neuro_u32 access = GENERIC_READ;
	neuro_u32 share = FILE_SHARE_READ;
	neuro_u32 create_diposition = OPEN_EXISTING;
	if (!m_bReadOnly)
	{
		access |= GENERIC_WRITE;
		create_diposition = bCreateAlways ? CREATE_ALWAYS : OPEN_ALWAYS;
		if (bShareWrite)
			share |= FILE_SHARE_WRITE;
	}

	m_hFile = CreateFile(util::StringUtil::MultiByteToWide(strFilePath).c_str(),
				access,          // open for writing
				share,                      // do not share
				NULL,                   // default security
				create_diposition,
				FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH ,// | FILE_FLAG_OVERLAPPED ,  // normal file
				NULL);                  // no attr. template

	if(m_hFile==INVALID_HANDLE_VALUE)
	{
		neuro_u32 dwErr=GetLastError();
		return false;
	}

	LARGE_INTEGER fileSize;
	if (!GetFileSizeEx(m_hFile, &fileSize))
		fileSize.QuadPart = 0;

	if (fileSize.QuadPart<init_size)
		fileSize.QuadPart = init_size;

    m_hFileMapping = CreateFileMapping(m_hFile, NULL,
											m_bReadOnly ? PAGE_READONLY : PAGE_READWRITE,    // 파일속성과 맞춤
											fileSize.HighPart,      // dwMaximumSizeHigh
											fileSize.LowPart,    // dwMaximumSizeLow
                                            NULL);
   
	m_nFileSize = fileSize.QuadPart;
	if(m_hFileMapping==NULL)
		return false;

	m_file_path=strFilePath;
	return true;
}

bool MMFDeviceAdaptor::SetPosition(neuro_u64 pos)
{
	m_position=pos;
	return true;
}

neuro_u32 MMFDeviceAdaptor::Read(void* buffer, neuro_u32 nSize) const
{
	neuro_u32 nTotalRead=0;
	__try
	{
		while(nSize>0)
		{
			neuro_u32 nReadable=0;
			neuro_u8* pbFile=((MMFDeviceAdaptor*)this)->GetMemoryBuffer(nSize, nReadable);
			if(!pbFile)
				return 0;

			if(nReadable>nSize)
				nReadable=nSize;

			memcpy(buffer, pbFile, nReadable);
			const_cast<MMFDeviceAdaptor*>(this)->m_position+=nReadable;
			nSize-=nReadable;

			nTotalRead+=nReadable;
		}
	}
	__except(GetExceptionCode()==EXCEPTION_IN_PAGE_ERROR ?	EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
	{
	// Failed to read from the view.
		nTotalRead=0;
	}

	return nTotalRead;
}

bool MMFDeviceAdaptor::SetUsageSize(neuro_u64 new_size)
{
	if(new_size<=m_nFileSize)
		return true;

	// 근데 이게 과연 잘 될까???
	if(m_pbFile)
		UnmapViewOfFile(m_pbFile);
	m_pbFile=NULL;

	m_nMappedPointer=0;
	m_nMappedSize=0;

	m_nFileSize=new_size;
	LARGE_INTEGER fileSize; fileSize.QuadPart = new_size;

	if(m_hFileMapping)
		CloseHandle(m_hFileMapping);
	m_hFileMapping = CreateFileMapping(m_hFile, NULL,
											m_bReadOnly ? PAGE_READONLY : PAGE_READWRITE,    // 파일속성과 맞춤
											fileSize.HighPart,      // dwMaximumSizeHigh
											fileSize.LowPart,    // dwMaximumSizeLow
											NULL);
	return m_hFileMapping != NULL;
}

neuro_u32 MMFDeviceAdaptor::Write(const void* buffer, neuro_u32 nSize)
{
	// 만약, 현재 파일 크기보다 쓰기해야할 공간이 더 크다면, 크기를 늘려야한다.
	if(m_position+nSize>m_nFileSize)
	{
		if(!SetUsageSize(m_position+nSize))
			return 0;
	}

	neuro_u32 nTotalWrite=0;
	__try
	{
		while(nTotalWrite<nSize)
		{
			neuro_u32 nWriteable=0;
			neuro_u8* pbFile=GetMemoryBuffer(nSize, nWriteable);
			if(!pbFile || nWriteable==0)
				return 0;

			if(nWriteable>nSize)
				nWriteable=nSize;

			memcpy(pbFile, buffer, nWriteable);
			m_position+=nWriteable;
			nSize-=nWriteable;

			nTotalWrite+=nWriteable;
		}
	}
	__except(GetExceptionCode()==EXCEPTION_IN_PAGE_ERROR ?	EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
	{
	// Failed to read from the view.
		nTotalWrite=0;
	}
	return nTotalWrite;
}

neuro_u8* MMFDeviceAdaptor::GetMemoryBuffer(neuro_u32 nSize, neuro_u32& nUsable)
{
	if(!m_hFileMapping)
		return NULL;

	if(!m_pbFile || m_position<m_nMappedPointer || m_position>=m_nMappedPointer+m_nMappedSize)
	{
		if(m_pbFile)
			UnmapViewOfFile(m_pbFile);

		LARGE_INTEGER newMappedPointer;

		newMappedPointer.QuadPart = m_position;
		// mapping 할땐 pointer는 m_dwAllocationGranularity 로 나누어 떨어져야 한다.
		newMappedPointer.LowPart = (newMappedPointer.LowPart / m_dwAllocationGranularity)*m_dwAllocationGranularity;

		if (m_nFileSize>0 && m_nFileSize<m_nMaxBlockSize)
			m_nMappedSize = (neuro_u32)m_nFileSize;
		else
			m_nMappedSize = m_nMaxBlockSize;

		if ((newMappedPointer.QuadPart + ((neuro_u64)m_nMappedSize)) > m_nFileSize)
			m_nMappedSize = m_nFileSize - newMappedPointer.QuadPart;

		neuro_u32 access = FILE_MAP_READ;
		if (!m_bReadOnly)
			access |= FILE_MAP_WRITE;
		m_pbFile = (PBYTE)MapViewOfFile(m_hFileMapping, access, newMappedPointer.HighPart, newMappedPointer.LowPart, m_nMappedSize);
		if (!m_pbFile)
		{
			m_nMappedPointer = 0;
			neuro_u32 dwError = GetLastError();
			return NULL;
		}

		m_nMappedPointer = newMappedPointer.QuadPart;
	}

	neuro_u32 nStartPos=m_position-m_nMappedPointer;

	nUsable=m_nMappedSize-nStartPos;

	return m_pbFile+nStartPos;
}

bool MMFDeviceAdaptor::Flush()
{
	return FlushViewOfFile(m_pbFile, m_nMappedSize);
}
