#include "stdafx.h"
#include "CacheReadBuffer.h"
#include "DeviceAdaptor.h"

using namespace ahnn::device;
using namespace ahnn;

CCacheReadBuffer::CCacheReadBuffer(DeviceAdaptor& stream, neuro_u32 nBufferSize)
: m_stream(stream)
{
	m_nStartPos=0;
	m_nDataSize=0;
	m_pBuffer=new unsigned __int8[nBufferSize];
	m_nBufferSize=nBufferSize;

	m_nEfficientBufferSize=nBufferSize;

	m_curDevicePos=0;

	InitializeCriticalSection(&m_cs);

	DWORD dwThreadID;
	m_hThread = CreateThread(NULL, 0, CacheReadThread, this, 0, &dwThreadID);

	m_hReadEvent = CreateEvent(NULL, TRUE, FALSE, _T("JNN.CacheRead.Event"));
	m_hStopEvent = CreateEvent(NULL, TRUE, FALSE, _T("JNN.CacheRead.Event"));
}

CCacheReadBuffer::~CCacheReadBuffer(void)
{
	SetEvent(m_hStopEvent);
	WaitForSingleObject(m_hThread, 1000);

	if(m_pBuffer)
		delete[] m_pBuffer;

	EnterCriticalSection(&m_cs);
	LeaveCriticalSection(&m_cs);

	DeleteCriticalSection(&m_cs);
}

neuro_u32 CCacheReadBuffer::ReadBuffer(neuro_pointer64 pos, unsigned __int8* buffer, neuro_u32 nSize) const
{
	if(nSize<=0)
		return 0;

	CCacheReadBuffer* _this=(CCacheReadBuffer*)this;

	EnterCriticalSection(&_this->m_cs);
	if(_this->m_nEfficientBufferSize<nSize)
		_this->m_nEfficientBufferSize=nSize*2;

	neuro_u32 nCacheRead=0;

	if(_this->m_curDevicePos!=pos)
	{
		if(_this->m_curDevicePos<pos && _this->m_nDataSize>pos - _this->m_curDevicePos)
		{
			_this->m_nStartPos += pos - _this->m_curDevicePos;
			_this->m_nDataSize -= pos - _this->m_curDevicePos;
		}
		else
		{
			_this->m_nStartPos=0;
			_this->m_nDataSize=0;
		}
		_this->m_curDevicePos=pos;
	}

	if(_this->m_nDataSize>0)
	{
		if(nSize<_this->m_nDataSize)
			nCacheRead=nSize;
		else
			nCacheRead=_this->m_nDataSize;

		memcpy(buffer, _this->m_pBuffer+_this->m_nStartPos, nCacheRead);

		_this->m_nStartPos+=nCacheRead;
		_this->m_nDataSize-=nCacheRead;

		buffer+=nCacheRead;
		nSize-=nCacheRead;

		_this->m_curDevicePos+=nCacheRead;
	}

	DWORD nAddReaded=0;
	if(nSize>0)
		nAddReaded=_this->m_stream.Read(_this->m_curDevicePos, buffer, nSize);

	LeaveCriticalSection(&_this->m_cs);

	SetEvent(_this->m_hReadEvent);

	// 다음 데이터를 paging에 넣기 위해 thread 로 실행시킨다.
	return nCacheRead + nAddReaded;
}

bool CCacheReadBuffer::Reallocate(neuro_u32 nNewBufferSize)
{
	EnterCriticalSection(&m_cs);
	if(nNewBufferSize<=m_nBufferSize)
	{
		LeaveCriticalSection(&m_cs);
		return true;
	}

	m_nBufferSize=nNewBufferSize;
	unsigned __int8* pNewBuffer=new unsigned __int8[m_nBufferSize];

	if(m_pBuffer)
	{
		if(m_nDataSize>0)
			memcpy(pNewBuffer, m_pBuffer+m_nStartPos, m_nDataSize);

		delete m_pBuffer;
	}

	m_pBuffer=pNewBuffer;
	m_nStartPos=0;

	LeaveCriticalSection(&m_cs);
	return m_pBuffer!=NULL;
}

DWORD WINAPI CCacheReadBuffer::CacheReadThread(LPVOID lpParam)
{
	if(lpParam==0)
		return 0;

	CCacheReadBuffer* _this=(CCacheReadBuffer*)lpParam;

	HANDLE phEvent[2]={_this->m_hStopEvent, _this->m_hReadEvent};
	while(WaitForMultipleObjects(2, phEvent, FALSE, INFINITE) != WAIT_OBJECT_0)
	{
		ResetEvent(_this->m_hReadEvent);

		EnterCriticalSection(&_this->m_cs);

		_this->Reallocate(_this->m_nEfficientBufferSize);

		neuro_u32 nWritePos=_this->m_nStartPos + _this->m_nDataSize;
		neuro_u32 nBufferSize=_this->m_nBufferSize - nWritePos;

		unsigned __int8* buffer=_this->m_pBuffer+nWritePos;

		neuro_u32 nReaded=_this->m_stream.Read(_this->m_curDevicePos, buffer, nBufferSize);
		if(nReaded>0)
		{
			_this->m_nDataSize+=nReaded;

			_this->m_curDevicePos+=nReaded;
		}

		LeaveCriticalSection(&_this->m_cs);;
	}
	return 0;
}
