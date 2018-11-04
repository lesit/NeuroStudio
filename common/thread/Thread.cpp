// Thread.cpp: implementation of the Thread class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "Thread.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

using namespace np::thread;

Thread::Thread()
{
	m_hThread=NULL;
}
      
Thread::~Thread()
{
	if(m_hThread)
		CloseHandle(m_hThread);
	m_hThread=NULL;
}

void Thread::Start()
{
	if(m_hThread)
		return;

	DWORD threadID = 0;

	m_hThread = CreateThread(NULL, 0, ThreadFunction, this, CREATE_SUSPENDED, &threadID);
	if(!m_hThread)
		return;

	::SetThreadPriority(m_hThread, THREAD_PRIORITY_NORMAL);
	::ResumeThread(m_hThread);
}

DWORD WINAPI Thread::ThreadFunction(void *pV)
{
	if(!pV)
		return 0;

	Thread* _this = (Thread*)pV;
	
	DWORD dwRet = _this->Run();

	Lock::Owner lock(_this->m_inst_lock);
	::CloseHandle(_this->m_hThread);	// 이렇게 해도 되나? WaitForSingleObject 중에도?
	_this->m_hThread = NULL;

	return dwRet;
}

bool Thread::IsRunning() const
{
	Lock::Owner lock(((Thread*)this)->m_inst_lock);
	
	return m_hThread != NULL;
}

void Thread::Wait()
{
	if (!Wait(INFINITE)) {};
}

bool Thread::Wait(DWORD timeoutMillis)
{
	if (!IsRunning())
		return true;

	bool bRet = ::WaitForSingleObject(m_hThread, timeoutMillis) == WAIT_OBJECT_0;

	if (!IsRunning())
		return true;

	return bRet;
}

void Thread::Terminate()
{
	if (!IsRunning())
		return;

	DWORD dwExit;
	if(GetExitCodeThread(m_hThread, &dwExit) && dwExit == STILL_ACTIVE)
	{
		// 원래는 dwExit을 넣어야 하지만, 무한대기하는 상태가 발생하는 경향이 있어서, 0으로 한다.
		if (!::TerminateThread(m_hThread, 0))	
			TerminateThread(m_hThread, -1);
	}
	if(m_hThread)
		CloseHandle(m_hThread);

	m_hThread=NULL;
}
