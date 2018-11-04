#if !defined(NEURO_DATA_SPEC_MODIFY_H)
#define NEURO_DATA_SPEC_MODIFY_H

#include "common.h"

#include "NeuroDataAllocSpec.h"
#include "NeuroStorageAllocationSystem.h"
#include "NeuroDataSpecTreeSearch.h"

namespace np
{
	namespace nsas
	{
		class NeuroDataSpecModify
		{
		public:
			NeuroDataSpecModify(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC &allocSpec);
			virtual ~NeuroDataSpecModify();

		protected:
			static bool m_is_verify;
			bool Verify(neuro_32 new_depth);

			const neuro_u32 m_pointersPerBlock;

			NeuroStorageAllocationSystem& m_nsas;
			_NEURO_DATA_ALLOC_SPEC& m_allocSpec;
		};
	}
}

#endif
