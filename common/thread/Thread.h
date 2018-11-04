// Thread.h: interface for the Thread class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(_NP_THREAD_H)
#define _NP_THREAD_H

#include <process.h>
#include "Lock.h"

namespace np
{
	namespace thread
	{
		class Thread
		{
		public:
			Thread();
			virtual ~Thread();

			void Start();

			bool IsRunning() const;

			void Wait();
			bool Wait(DWORD timeoutMillis);

			void Terminate();

		private:
			virtual int Run() = 0;

			static DWORD WINAPI ThreadFunction(void *pV);

			HANDLE m_hThread;

			Lock m_inst_lock;
		};
	};
}

#endif
