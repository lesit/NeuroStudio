#if !defined(_NEURO_DATA_SPEC_TREE_SEARCH)
#define _NEURO_DATA_SPEC_TREE_SEARCH

#include "common.h"

#include "NeuroDataAllocSpec.h"
#include "NeuroStorageAllocationSystem.h"

namespace np
{
	namespace nsas
	{
		class Pointer_Table_Spec_Base
		{
		public:
			Pointer_Table_Spec_Base()
			{
				search_node_scope = 0;
				child = NULL;
			}
			virtual ~Pointer_Table_Spec_Base(){};

			virtual Pointer_Table_Spec_Base* GetParent(){ return NULL; }

			const Pointer_Table_Spec_Base* GetParent() const{ return const_cast<Pointer_Table_Spec_Base*>(this)->GetParent(); }

			virtual neuro_block GetTableBlockNo() const { return 0; }
			virtual neuro_u8* GetBlockData() { return NULL; }

			virtual _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* GetPointerNodeList() = 0;

			virtual neuro_u32 GetPointerNodeCount() = 0;
			virtual void SetPointerNodeCount(neuro_u32 n) = 0;

			virtual neuro_block GetNextPointerTable() const { return 0; }
			virtual void ChangeNextPointerTable(neuro_block no){}

			neuro_u32 search_node_scope;

			Pointer_Table_Spec_Base* child;
		};

		class Root_Spec : public Pointer_Table_Spec_Base
		{
		public:

			Root_Spec(_NEURO_DATA_ALLOC_SPEC& allocSpec)
				: m_allocSpec(allocSpec)
			{}
			virtual ~Root_Spec(){};

			_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* GetPointerNodeList() override { return m_allocSpec.block_bptree_root; }
			neuro_u32 GetPointerNodeCount() override { return m_allocSpec.node_count; }
			void SetPointerNodeCount(neuro_u32 n) override { m_allocSpec.node_count = n; }

			_NEURO_DATA_ALLOC_SPEC& m_allocSpec;
		};

		class Pointer_Table_Spec : public Pointer_Table_Spec_Base
		{
		public:
			Pointer_Table_Spec(Pointer_Table_Spec_Base* pParent)
			{
				this->parent = pParent;
				if (pParent)
					pParent->child = this;

				this->block_no = 0;
				pt_info.block_data = NULL;
			}

			virtual ~Pointer_Table_Spec()
			{
				if (pt_info.block_data)
					free(pt_info.block_data);

				delete parent;
			}

			bool ReadTable(NeuroStorageAllocationSystem& nsas, neuro_block block_no)
			{
				if (pt_info.block_data == NULL)
					pt_info.block_data = (neuro_u8*)malloc(nsas.GetBlockSize());

				if (!nsas.BlockIO(block_no, pt_info.block_data, false))
				{
					DEBUG_OUTPUT(L"failed to read block[%llu]", block_no);
					free(pt_info.block_data);
					pt_info.block_data = NULL;
					return false;
				}
				this->block_no = block_no;
				return true;
			}

			Pointer_Table_Spec_Base* GetParent() override { return parent; }

			neuro_block GetTableBlockNo() const override { return block_no; }
			neuro_u8* GetBlockData() override { return pt_info.block_data; }

			_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* GetPointerNodeList() override { return pt_info.GetPointerNodeList(); }
			neuro_u32 GetPointerNodeCount() override { return pt_info.GetHeader()->node_count; }
			void SetPointerNodeCount(neuro_u32 n) override { pt_info.GetHeader()->node_count = n; }

			neuro_block GetNextPointerTable() const override { return pt_info.GetHeader()->next_pointer_table; }
			virtual void ChangeNextPointerTable(neuro_block no){ pt_info.GetHeader()->next_pointer_table = no; }

			Pointer_Table_Spec_Base* parent;

			neuro_block block_no;
			_POINTER_TABLE_INFO pt_info;
		};

		struct Pointer_Table_Spec_Path
		{
			Pointer_Table_Spec_Path()
			{
				total_depth = 0;

				root = NULL;
				leaf = NULL;
			}
			~Pointer_Table_Spec_Path()
			{
				delete leaf;
			}

			neuro_32 total_depth;

			Pointer_Table_Spec_Base* root;
			Pointer_Table_Spec_Base* leaf;
		};

		class NeuroDataSpecTreeSearch
		{
		public:
			NeuroDataSpecTreeSearch(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC &allocSpec);
			virtual ~NeuroDataSpecTreeSearch();
			bool CreatePath(neuro_u64 size, Pointer_Table_Spec_Path& ret) const;

			struct _NDA_TREE_INFO
			{
				neuro_32 depth;
				neuro_u64 size_per_root_node;
			};
			neuro_32 GetDepthInfo(neuro_u64 size, _NDA_TREE_INFO& info) const;

		private:
			NeuroStorageAllocationSystem& m_nsas;
			_NEURO_DATA_ALLOC_SPEC& m_allocSpec;

			const neuro_u32 m_pointersPerBlock;
		};
	}
}
#endif
