#include "stdafx.h"

#include "DataPreloadBuffer.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

DataPreloadBuffer::DataPreloadBuffer()
{
	m_start_pos = 0;

	m_total_data = 0;
	m_value_bytes = 0;
	m_buffer = NULL;
}

DataPreloadBuffer::~DataPreloadBuffer()
{
	Clear();
}

void DataPreloadBuffer::Clear()
{
	if(m_buffer)
		free(m_buffer);
	m_total_data = 0;
}

bool DataPreloadBuffer::Resize(neuro_u64 count)
{
	if (m_value_bytes == 0)
		return false;

	m_buffer = realloc(m_buffer, count * m_value_bytes);
	if (m_buffer)
		m_total_data = count;

	return m_buffer != NULL;
}

bool DataPreloadBuffer::Set(neuro_u64 pos, void* buf)
{
	void* preload_buffer = Get(pos);
	if (!preload_buffer)
		return false;
	
	memcpy(preload_buffer, buf, m_value_bytes);
	return true;
}

void* DataPreloadBuffer::Get(neuro_u64 pos)
{
	if (m_buffer == NULL)
		return NULL;

	neuro_u64 index = pos - m_start_pos;
	if (index >= m_total_data)
		return NULL;

	return (neuro_u8*)m_buffer + index*m_value_bytes;
}

bool DataPreloadBuffer::Read(neuro_u64 pos, void* buf)
{
	if (m_total_data == 0)
		return false;

	void* preload_buffer = Get(pos);
	if (!preload_buffer)
		return false;

	memcpy(buf, preload_buffer, m_value_bytes);
	return true;
}

