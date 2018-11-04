#pragma once

#include "../Common/JNM_Common.h"
namespace ahnn
{
	namespace device
	{
		class DeviceAdaptor;
		class CCacheReadBuffer
		{
		public:
			CCacheReadBuffer(DeviceAdaptor& stream, neuro_u32 nBufferSize=1024);
			~CCacheReadBuffer(void);

			neuro_u32 ReadBuffer(neuro_pointer64 pos, unsigned __int8* buffer, neuro_u32 nSize) const;

		protected:
			bool Reallocate(neuro_u32 nNewBufferSize);

		private:
			unsigned __int8* m_pBuffer;
			neuro_u32 m_nBufferSize;
			neuro_pointer32 m_nStartPos;
			neuro_u32 m_nDataSize;

			neuro_u32 m_nEfficientBufferSize;

			DeviceAdaptor& m_stream;
			neuro_pointer64 m_curDevicePos;

			static DWORD WINAPI CacheReadThread(LPVOID lpParam);
		private:
			CRITICAL_SECTION m_cs;

			HANDLE m_hThread;
			HANDLE m_hReadEvent;
			HANDLE m_hStopEvent;
		};
	}
}
