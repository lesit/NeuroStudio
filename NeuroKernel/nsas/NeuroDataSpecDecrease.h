#if !defined(NEURO_DATA_SPEC_DECREASE_H)
#define NEURO_DATA_SPEC_DECREASE_H

#include "NeuroDataSpecModify.h"

namespace np
{
	namespace nsas
	{
		class NeuroStorageAllocationSystem;

		class NeuroDataSpecDecrease : public NeuroDataSpecModify
		{
		public:
			NeuroDataSpecDecrease(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec);
			virtual ~NeuroDataSpecDecrease();

			bool DeleteAll();
			bool Decrease(neuro_u64 new_size);

		protected:
			neuro_block DeleteChildBlocks(neuro_u32 depth, Pointer_Table_Spec_Base& pt, neuro_block start_child);

			bool DeleteBlocks(neuro_u64 start, neuro_u64 count);
		};
	}
}

#endif
