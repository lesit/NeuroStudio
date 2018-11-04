#include "stdafx.h"

#include "CacheWriteBuffer.h"
#include "DeviceAdaptor.h"

using namespace ahnn;
using namespace ahnn::device;

CCacheWriteBuffer::CCacheWriteBuffer(DeviceAdaptor& stream)
: m_stream(stream)
{
	InitializeCriticalSection(&m_csWrite);
}

CCacheWriteBuffer::~CCacheWriteBuffer()
{
	DeleteCriticalSection(&m_csWrite);
}

neuro_u32 CCacheWriteBuffer::WriteBuffer(neuro_pointer64 pos, const unsigned __int8* buffer, neuro_u32 nBytes)
{
	_WRITE_DATA* write=new _WRITE_DATA;
	write->cs=&m_csWrite;
	write->stream=&m_stream;
	write->pos=pos;
	write->buffer=buffer;
	write->nBytes=nBytes;

	EnterCriticalSection(&m_csWrite);
	LeaveCriticalSection(&m_csWrite);

	DWORD dwThreadID;
	HANDLE hThread = CreateThread(NULL, 0, WriteThread, write, 0, &dwThreadID);
	return 0;
}

DWORD WINAPI CCacheWriteBuffer::WriteThread(LPVOID lpParam)
{
	_WRITE_DATA* write=(_WRITE_DATA*)lpParam;
	if(!write)
		return 0;

	EnterCriticalSection(write->cs);

	write->stream->Write(write->pos, write->buffer, write->nBytes);
	delete write;

	LeaveCriticalSection(write->cs);

	delete write;
	return 0;
}
