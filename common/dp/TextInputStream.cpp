#include <locale.h>
#include <io.h>
#include <fcntl.h>

#include "TextInputStream.h"

#include "../util/np_util.h"
#include "../util/StringUtil.h"

using namespace np::dp;
using namespace np::util;

TextFileInputStream::TextFileInputStream()
{
	m_start_position = 0;
	m_type = _textfile_type::ansi;

	m_read_buffer = NULL;
	m_input_device = NULL;
}

TextFileInputStream::~TextFileInputStream()
{
	if (m_read_buffer)
		free(m_read_buffer);
}

#include <tchar.h>

bool TextFileInputStream::SetInputDevice(device::DeviceAdaptor* device)
{
	if (device == NULL)
		return false;

	// ANSI 문자열을 읽어 들일때 2byte의 한글어, 중국어, 일본어등을 자동으로 변환하기 위해 설정한다.
	wchar_t* pLocale = _wsetlocale(LC_ALL, L"");
	unsigned char first_4bytes[4];
	int bom_count = device->Read(first_4bytes, sizeof(first_4bytes));
	if(bom_count<=0)
		return false;

	int bomlen;
	m_type = CheckTextMode(first_4bytes, bom_count, bomlen);
	if (m_type!=_textfile_type::ansi && m_type!=_textfile_type::utf8 && m_type!=_textfile_type::utf16le)
	{
		DEBUG_OUTPUT(L"can't read type text");
		return false;
	}
	if (m_type == _textfile_type::ansi)	// bom이 없는 utf 일지 모르기 때문에 한번더 확인
	{
		device->SetPosition(0);
		unsigned char text[1024];
		int read = device->Read(text, sizeof(text));
		
		if (StringUtil::CheckUtf8(text, read) == _str_utf8_type::utf8)
			m_type = _textfile_type::utf8;
	}

	m_start_position = bomlen;
	device->SetPosition(m_start_position);

	m_input_device = device;
	return true;
}

TextFileInputStream::_textfile_type TextFileInputStream::CheckTextMode(unsigned char* first_4bytes, int size, int& bomlen)
{
	bomlen = 0;
	if (size>=3 && first_4bytes[0] == 0xEF && first_4bytes[1] == 0xBB && first_4bytes[2] == 0xBF)
	{	// utf-8
		bomlen = 3;
		return _textfile_type::utf8;
	}
	else if (size >= 2 && first_4bytes[0] == 0xFF && first_4bytes[1] == 0xFE)
	{	// utf16le
		bomlen = 2;
		return _textfile_type::utf16le;
	}
	else if (size >= 2 && first_4bytes[0] == 0xFE && first_4bytes[1] == 0xFF)
	{
		bomlen = 2;
		return _textfile_type::utf16be;
	}
	else if (size >= 4 && first_4bytes[0] == 0x00 && first_4bytes[1] == 0x00 && first_4bytes[2] == 0xFE && first_4bytes[3] == 0xFF)
	{	// utf32BE
		bomlen = 4;
		return _textfile_type::utf32be;
	}
	else if (size >= 4 && first_4bytes[0] == 0xFF && first_4bytes[1] == 0xFE && first_4bytes[2] == 0x00 && first_4bytes[3] == 0x00)
	{	// UTF - 32LE 
		bomlen = 4;
		return _textfile_type::utf32le;
	}
	else
	{

	}
	return _textfile_type::ansi;
}

neuro_u8 TextFileInputStream::ReadCharSize() const
{
	switch (m_type)
	{
	case _textfile_type::ansi:
	case _textfile_type::utf8:
		return 1;
	case _textfile_type::utf16le:
	case _textfile_type::utf16be:
		return 2;
	case _textfile_type::utf32le:
	case _textfile_type::utf32be:
		return 4;
	}
	return 0;
}

neuro_64 TextFileInputStream::ReadString(wchar_t* buffer, neuro_64 count)
{
	if (m_input_device == NULL)
		return false;

	switch (m_type)
	{
	case _textfile_type::utf16be:
	case _textfile_type::utf32le:
	case _textfile_type::utf32be:
		return 0;
	}

	if (m_type == _textfile_type::utf16le)
		return m_input_device->Read(buffer, count);

	// ansi, utf8
	if (m_read_buffer)
		m_read_buffer = (char*)realloc(m_read_buffer, count);
	else
		m_read_buffer = (char*)malloc(count);

	neuro_64 read = m_input_device->Read(m_read_buffer, count);
	if (read == 0)
		return 0;

	const char* multibyte_first = (const char*)StringUtil::FindLastTruncatedMultibytesBegin((unsigned char*)m_read_buffer, read);
	if (multibyte_first == NULL)
		return 0;

	int multi_need = m_read_buffer + read - multibyte_first;
	if (multi_need > 0)
	{
		m_input_device->SetPosition(m_input_device->GetPosition() - multi_need);
		read -= multi_need;
	}

	return StringUtil::MultiByteToWide(m_read_buffer, read, buffer, count, m_type == _textfile_type::utf8);
}

wchar_t* TextFileInputStream::ReadStringAllWide(int& size)
{
	if (m_input_device == NULL)
		return NULL;

	switch (m_type)
	{
	case _textfile_type::utf16be:
	case _textfile_type::utf32le:
	case _textfile_type::utf32be:
		return NULL;
	}

	MoveDataPos(0);

	neuro_size_t file_size = m_input_device->GetUsageSize() - m_start_position;
	if (m_type == _textfile_type::utf16le)
	{
		wchar_t* ret = new wchar_t[file_size + 1];
		size = m_input_device->Read(ret, file_size) / 2;
		if (size == 0)
		{
			delete[] ret;
			return NULL;
		}
		ret[size] = L'0';
		return ret;
	}
	else if (m_type == _textfile_type::ansi || m_type == _textfile_type::utf8)
	{
		// ansi, utf8
		if (m_read_buffer)
			m_read_buffer = (char*)realloc(m_read_buffer, file_size);
		else
			m_read_buffer = (char*)malloc(file_size);

		neuro_64 read = m_input_device->Read(m_read_buffer, file_size);
		if (read == 0)
			return 0;

		return StringUtil::MultiByteToWide(m_read_buffer, read, size, m_type == _textfile_type::utf8);
	}
	else
		return NULL;
}

char* TextFileInputStream::ReadStringAllUtf8(int& size)
{
	if (m_input_device == NULL)
		return NULL;

	switch (m_type)
	{
	case _textfile_type::utf16be:
	case _textfile_type::utf32le:
	case _textfile_type::utf32be:
		return NULL;
	}

	MoveDataPos(0);

	neuro_size_t file_size = m_input_device->GetUsageSize() - m_start_position;

	if (m_type == _textfile_type::ansi || m_type == _textfile_type::utf16le)
	{
		const int ch_size = m_type == _textfile_type::utf16le ? sizeof(wchar_t) : sizeof(char);

		if (m_read_buffer)
			m_read_buffer = (char*)realloc(m_read_buffer, file_size + ch_size);
		else
			m_read_buffer = (char*)malloc(file_size + ch_size);

		int read = m_input_device->Read(m_read_buffer, file_size);
		if (read <= 0)
			return NULL;

		m_read_buffer[file_size] = 0;
		if (ch_size>1)
			m_read_buffer[file_size+1] = 0;

		if (m_type == _textfile_type::ansi)
		{
			int wide_size;
			wchar_t* wide = StringUtil::MultiByteToWide(m_read_buffer, read, wide_size, false);
			if (wide == NULL)
				return NULL;

			char* ret = StringUtil::WideToMultiByte(wide, wide_size, size);
			delete[] wide;
			return ret;
		}
		if (m_type == _textfile_type::utf16le)
			return StringUtil::WideToMultiByte((const wchar_t*)m_read_buffer, -1, size);
	}
	else if (m_type == _textfile_type::utf8)
	{
		char* ret = (char*)malloc(file_size + 1);
		size = m_input_device->Read(ret, file_size);
		if (size <= 0)
			return 0;

		ret[size] = '\0';
		return ret;
	}

	return NULL;
}

neuro_64 TextFileInputStream::GetCurrentDataPos() const
{
	return m_input_device->GetPosition() - m_start_position;
}

bool TextFileInputStream::MoveDataPos(neuro_64 pos)
{
	return m_input_device->SetPosition(pos + m_start_position);
}
