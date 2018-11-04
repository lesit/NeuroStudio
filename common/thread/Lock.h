// Locking.h: interface for the CLocking class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(_NP_LOCK_H)
#define _NP_LOCK_H

#if defined(WIN32) | defined(WIN64)
#include <windows.h>
#endif

namespace np
{
	namespace thread
	{
		class Lock  
		{
		public :
			class Owner
			{
			public:
				explicit Owner(Lock &crit);
				virtual ~Owner();

			private :
				Lock &m_crit;
			};

			Lock();
			virtual ~Lock();
			void Enter();
			void Leave();

		private :
			CRITICAL_SECTION m_crit;
		};
	};
}

#endif // !defined(AFX_LOCKING_H__2ED1F38E_854D_4DA9_8E1D_7556BB7F9158__INCLUDED_)
