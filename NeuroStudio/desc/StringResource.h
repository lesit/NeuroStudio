#if !defined(_STRING_RESOURCE_H)
#define _STRING_RESOURCE_H

#include "common.h"

namespace np
{
	namespace str_rc
	{
		class StringResource
		{
		public:
			static std::wstring GetString(unsigned int id);

			StringResource();
			virtual ~StringResource();
		};
	}
}

using namespace np::str_rc;

#endif
