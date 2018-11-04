#include "stdafx.h"
#include "NeuroDataSpecIncrease.h"

#include "NeuroStorageAllocationSystem.h"

#include "NeuroDataAccessManager.h"

using namespace np;
using namespace np::device;
using namespace np::nsas;

NeuroDataSpecIncrease::NeuroDataSpecIncrease(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec)
: NeuroDataSpecModify(nsas, allocSpec)
{
	memset(&m_new_pt_buffer, 0, sizeof(_POINTER_TABLE_INFO));
	m_new_pt_buffer.block_data = (neuro_u8*)malloc(m_nsas.GetBlockSize());

	m_new_depth = 0;

	m_last_alloc_block = 0;
}

NeuroDataSpecIncrease::~NeuroDataSpecIncrease()
{
	free(m_new_pt_buffer.block_data);
}

bool NeuroDataSpecIncrease::Increase(neuro_u64 new_size)
{
	if (new_size <= m_allocSpec.size)
		return true;

#ifdef _DEBUG
	if (new_size > m_nsas.GetBlockSize())	// node count가 잘못 계산된다! ㅡㅡ
		int a = 0;
#endif

	neuro_u64 prev_block = NP_Util::CalculateCountPer(m_allocSpec.size, m_nsas.GetBlockSize());
	neuro_u64 new_block = NP_Util::CalculateCountPer(new_size, m_nsas.GetBlockSize());

	neuro_u32 extend_block = static_cast<neuro_u32>(new_block - prev_block);
	if (extend_block == 0)
	{
		m_allocSpec.size = new_size;
		return true;
	}

#ifdef _DEBUG
	if (new_size == 64000)
		int a = 0;
#endif

	Pointer_Table_Spec_Path pt_spec_path;
	NeuroDataSpecTreeSearch nds_tree(m_nsas, m_allocSpec);
	if (!nds_tree.CreatePath(m_allocSpec.size, pt_spec_path))
	{
		DEBUG_OUTPUT(L"failed create path");
		return false;
	}

	_DYNMEM_PTLIST_INFO datablock_pt_list(extend_block);
	if (!datablock_pt_list.list)
	{
		DEBUG_OUTPUT(L"no new pointer table list");
		return false;
	}

	DEBUG_OUTPUT(L"AllocBlock(%u) for pointer list", extend_block);
	neuro_block relative_block = pt_spec_path.leaf->GetTableBlockNo();
	if(pt_spec_path.leaf->GetPointerNodeCount()>0)
		relative_block = pt_spec_path.leaf->GetPointerNodeList()[pt_spec_path.leaf->GetPointerNodeCount() - 1].block_no;

	if (m_nsas.AllocBlocks(relative_block, datablock_pt_list.count, datablock_pt_list.list) == 0)
	{
		DEBUG_OUTPUT(L"fail AllocBlock");
		return false;
	}
	m_last_alloc_block = datablock_pt_list.list[datablock_pt_list.count - 1].block_no;

#ifdef _DEBUG
	if (new_size == 64000)
		int a = 0;
#endif

	m_new_depth = pt_spec_path.total_depth;

	// 이제 이 data block들을 트리에 삽입하자!
	if (!InsertBlocksIntoTable(datablock_pt_list, pt_spec_path.leaf))
		return false;

	m_allocSpec.size = new_size;

	// 검증을 해보자!!
	if (!Verify(m_new_depth))
	{
		DEBUG_OUTPUT(L"failed verify");
		return false;
	}

	return true;
}

bool NeuroDataSpecIncrease::InsertBlocksIntoTable(_DYNMEM_PTLIST_INFO& pt_list, Pointer_Table_Spec_Base* cur_pt_spec)
{
	if (pt_list.count == 0)
		return true;

	DEBUG_OUTPUT(L"insert %u pointers", pt_list.count);

	if (cur_pt_spec->GetParent() == NULL)	// 만약 root인 경우
	{
		DEBUG_OUTPUT(L"old root node is %u", m_allocSpec.node_count);

#ifdef _DEBUG
		if (pt_list.count == 98)
			int a = 0;
#endif

		// 만약 root인 경우, 새로 만든 pointer list를 root에 다 넣을수 있을때까지 반복시킨다.
		while (cur_pt_spec->GetPointerNodeCount() + pt_list.count> g_alloc_btree_root_count)	// 만약 root에 다 넣지 못할 경우
		{
			DEBUG_OUTPUT(L"old %u root. new %u pointer tables is over root[%u]. try new parent again", cur_pt_spec->GetPointerNodeCount(), pt_list.count, g_alloc_btree_root_count);

			// 둘다 넣은 새로운 리스를 만든다.
			_DYNMEM_PTLIST_INFO new_pt_list;
			new_pt_list.Alloc(cur_pt_spec->GetPointerNodeCount() + pt_list.count);
			memcpy(new_pt_list.list, cur_pt_spec->GetPointerNodeList(), cur_pt_spec->GetPointerNodeCount()*sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));
			memcpy(new_pt_list.list + cur_pt_spec->GetPointerNodeCount(), pt_list.list, pt_list.count*sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));

			_DYNMEM_PTLIST_INFO parent_pt_list;	// 이것을 넣기 위한 부모를 만들고 넣는다.
			if (!CreateParentPointerTableList(new_pt_list.count, new_pt_list.list, parent_pt_list))
			{
				DEBUG_OUTPUT(L"failed CreateParentPointerTableList. %u pointers", new_pt_list.count);
				return false;
			}

			free(pt_list.list);

			pt_list.count = parent_pt_list.count;
			pt_list.list = parent_pt_list.list;
			m_allocSpec.node_count = 0;

			parent_pt_list.list = NULL;

			++m_new_depth;
		}
		
		memcpy(m_allocSpec.block_bptree_root + m_allocSpec.node_count, pt_list.list, pt_list.count*sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));
		m_allocSpec.node_count += pt_list.count;

		DEBUG_OUTPUT(L"new root node is %u", m_allocSpec.node_count);
		return true;
	}

	neuro_u32 copy_count = min(m_pointersPerBlock - cur_pt_spec->search_node_scope, pt_list.count);
	memcpy(cur_pt_spec->GetPointerNodeList() + cur_pt_spec->search_node_scope, pt_list.list, copy_count * sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));

	_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* list = pt_list.list + copy_count;

	cur_pt_spec->SetPointerNodeCount(cur_pt_spec->search_node_scope + copy_count);

	copy_count = pt_list.count - copy_count;
	if (copy_count == 0)// 다했음
	{
		if (!m_nsas.BlockIO(cur_pt_spec->GetTableBlockNo(), cur_pt_spec->GetBlockData(), true))
		{
			DEBUG_OUTPUT(L"failed write pointer table block[%llu]", cur_pt_spec->GetTableBlockNo());
			return false;
		}
		return true;
	}

	// 아직 더 남았다. 새로운 다음 포인터 테이블 블록들을 생성하고, 그 테이블 블록에 이 남은 포인터 리스트를 넣어야 한다.
	_DYNMEM_PTLIST_INFO next_pt_list;
	if (!CreateParentPointerTableList(copy_count, list, next_pt_list))
	{
		DEBUG_OUTPUT(L"failed CreateParentPointerTableList. %u pointers", copy_count);
		return false;
	}

	cur_pt_spec->ChangeNextPointerTable(next_pt_list.list[0].block_no);
	if (!m_nsas.BlockIO(cur_pt_spec->GetTableBlockNo(), cur_pt_spec->GetBlockData(), true))
	{
		DEBUG_OUTPUT(L"failed write previous pointer table block[%llu]", cur_pt_spec->GetTableBlockNo());
		return false;
	}
	// 새로 만든 포인터 테이블 블록은 부모 포인터 테이블에 넣어야 한다.
	return InsertBlocksIntoTable(next_pt_list, cur_pt_spec->GetParent());
}

bool NeuroDataSpecIncrease::CreateParentPointerTableList(neuro_u32 count, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* list, _DYNMEM_PTLIST_INFO& parent_pt_list)
{
	if (count == 0)
	{
		DEBUG_OUTPUT(L"failed. count is zero.");
		return false;
	}
	DEBUG_OUTPUT(L"make new parent for %u pointers", count);

	parent_pt_list.Alloc(NP_Util::CalculateCountPer(count, m_pointersPerBlock));

	if (m_nsas.AllocBlocks(m_last_alloc_block, parent_pt_list.count, parent_pt_list.list) == 0)
	{
		DEBUG_OUTPUT(L"fail AllocBlock");
		return false;
	}
	m_last_alloc_block = parent_pt_list.list[parent_pt_list.count - 1].block_no;

	DEBUG_OUTPUT(L"try to create %u parent pointer tables", parent_pt_list.count);
	for (neuro_u32 i = 0; i < parent_pt_list.count; i++)
	{
		neuro_u32 node_count = min(count, m_pointersPerBlock);
		memcpy(m_new_pt_buffer.GetPointerNodeList(), list, node_count * sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));

		m_new_pt_buffer.GetHeader()->node_count = node_count;

		if (i + 1 < parent_pt_list.count)
			m_new_pt_buffer.GetHeader()->next_pointer_table = parent_pt_list.list[i + 1].block_no;
		else
			m_new_pt_buffer.GetHeader()->next_pointer_table = 0;

		if (!m_nsas.BlockIO(parent_pt_list.list[i].block_no, (neuro_u8*)m_new_pt_buffer.block_data, true))
		{
			DEBUG_OUTPUT(L"failed write new pointer table block[%llu]", parent_pt_list.list[i].block_no);
			return false;
		}
		DEBUG_OUTPUT(L"created parent pointer table[%llu]. next table[%llu]", parent_pt_list.list[i].block_no, m_new_pt_buffer.GetHeader()->next_pointer_table);

		list += node_count;
		count -= node_count;
	}

	return true;
}
