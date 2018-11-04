#include "Lock.h"

using namespace np::thread;

Lock::Owner::Owner(Lock &crit)
: m_crit(crit)
{
	m_crit.Enter();
}

Lock::Owner::~Owner()
{
   m_crit.Leave();
}

Lock::Lock()
{
   ::InitializeCriticalSection(&m_crit);
}
				  

Lock::~Lock()
{
	::DeleteCriticalSection(&m_crit);
}

void Lock::Enter()
{
	::EnterCriticalSection(&m_crit);
}

void Lock::Leave()
{
	::LeaveCriticalSection(&m_crit);
}
