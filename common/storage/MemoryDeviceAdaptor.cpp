#include "MemoryDeviceAdaptor.h"

#include "util/np_util.h"

using namespace np::device;

MemoryDeviceRefAdaptor::MemoryDeviceRefAdaptor(neuro_u32 size, void* reference)
{
	m_buf_size = size;
	m_buffer = reference;
	m_position = 0;
}

MemoryDeviceRefAdaptor::MemoryDeviceRefAdaptor()
{
	m_buffer = NULL;
	m_buf_size = 0;
	m_position = 0;
}

bool MemoryDeviceRefAdaptor::SetPosition(neuro_u64 pos)
{
	m_position = pos;
	return true;
}

neuro_u32 MemoryDeviceRefAdaptor::Read(void* buffer, neuro_u32 size) const
{
	if (m_position >= m_buf_size)
		return 0;

	if (m_position + size>m_buf_size)
		size = m_buf_size - m_position;

	memcpy(buffer, (neuro_u8*)m_buffer + m_position, size);
	const_cast<MemoryDeviceRefAdaptor*>(this)->m_position += size;
	return size;
}

MemoryDeviceAdaptor::MemoryDeviceAdaptor(neuro_u64 init_size)
{
	if (init_size)
	{
		m_buffer = malloc(init_size);
		if (m_buffer)
			m_buf_size = init_size;
	}

	m_position=0;
}

MemoryDeviceAdaptor::~MemoryDeviceAdaptor()
{
	if (m_buffer)
		free(m_buffer);
}

neuro_u32 MemoryDeviceAdaptor::Write(const void* buffer, neuro_u32 size)
{
	if (size == 0)
		return 0;

	if (m_buf_size<m_position)
		m_position = m_buf_size;

	neuro_u64 need = size - (m_buf_size - m_position);
	if (need > 0)
	{
		need = NP_Util::CalculateCountPer(need, 1024) * 1024;	// 최소 1 kb 단위로 할당하도록
		if (m_buffer == NULL)
			m_buffer = (neuro_u8*)malloc(m_buf_size + need);
		else
			m_buffer = (neuro_u8*)realloc(m_buffer, m_buf_size + need);
		if (!m_buffer)
		{
			m_buf_size = 0;
			m_position = 0;
			return false;
		}
		m_buf_size += need;
	}

	memcpy((neuro_u8*)m_buffer + m_position, buffer, size);
	m_position += size;
	return size;
}
