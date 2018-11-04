#pragma once

#include "../storage/FileDeviceAdaptor.h"

namespace np
{
	namespace dp
	{
		class TextFileInputStream
		{
		public:
			TextFileInputStream();
			virtual ~TextFileInputStream();

			bool SetInputDevice(device::DeviceAdaptor* device);

			enum class _textfile_type{ansi, utf8, utf16le, utf16be, utf32le, utf32be};
			static _textfile_type CheckTextMode(unsigned char* first_4bytes, int size, int& bomlen);

			neuro_64 ReadString(wchar_t* buffer, neuro_64 count);
			wchar_t* ReadStringAllWide(int& size);
			char* ReadStringAllUtf8(int& size);

			neuro_64 GetCurrentDataPos() const;
			bool MoveDataPos(neuro_64);

			neuro_u8 ReadCharSize() const;
		protected:
			_textfile_type m_type;
			neuro_u64 m_start_position;

			char* m_read_buffer;

		protected:
			device::DeviceAdaptor* m_input_device;
		};
	}
}
