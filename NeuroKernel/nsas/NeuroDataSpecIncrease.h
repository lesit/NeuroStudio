#if !defined(NEURO_DATA_SPEC_INCREASE_H)
#define NEURO_DATA_SPEC_INCREASE_H

#include "NeuroDataSpecModify.h"

namespace np
{
	namespace nsas
	{
		class NeuroStorageAllocationSystem;

		struct _DYNMEM_PTLIST_INFO
		{
			_DYNMEM_PTLIST_INFO(neuro_u32 n = 0)
			{
				list = NULL;
				Alloc(n);
			}
			~_DYNMEM_PTLIST_INFO()
			{
				if (list)
					free(list);
			}
			bool Alloc(neuro_u32 n)
			{
				count = n;
				if (n == 0)
					return true;

				if (list)
					list = (_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE*)realloc(list, sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE)*n);
				else
					list = (_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE*)malloc(sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE)*n);
				if (list != NULL)
					return true;

				count = 0;
				return false;
			}
			_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* list;
			neuro_u32 count;
		};
		class NeuroDataSpecIncrease : public NeuroDataSpecModify
		{
		public:
			NeuroDataSpecIncrease(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec);
			virtual ~NeuroDataSpecIncrease();

			bool Increase(neuro_u64 new_size);

		protected:
			bool InsertBlocksIntoTable(_DYNMEM_PTLIST_INFO& pt_list, Pointer_Table_Spec_Base* cur_pt_spec);
			bool CreateParentPointerTableList(neuro_u32 count, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* list, _DYNMEM_PTLIST_INFO& parent_pt_list);

			_POINTER_TABLE_INFO m_new_pt_buffer;

			neuro_32 m_new_depth;

			neuro_block m_last_alloc_block;
		};
	}
}

#endif
