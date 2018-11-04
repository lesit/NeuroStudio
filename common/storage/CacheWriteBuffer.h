#if !defined(_CACHE_STREAM_WRITE_BUFFER_H)
#define _CACHE_STREAM_WRITE_BUFFER_H

#include "../Common/JNM_Common.h"

namespace ahnn
{
	namespace device
	{
		class DeviceAdaptor;

		class CCacheWriteBuffer
		{
		public:
			CCacheWriteBuffer(DeviceAdaptor& stream);
			~CCacheWriteBuffer();

			neuro_u32 WriteBuffer(neuro_pointer64 pos, const unsigned __int8* buffer, neuro_u32 nBytes);

		private:
			struct _WRITE_DATA
			{
				CRITICAL_SECTION* cs;

				device::DeviceAdaptor* stream;
				neuro_pointer64 pos;
				const unsigned __int8* buffer;
				neuro_u32 nBytes;
			};
			static DWORD WINAPI WriteThread(LPVOID lpParam);

			DeviceAdaptor& m_stream;
			CRITICAL_SECTION m_csWrite;	
		};
	}
}

#endif
