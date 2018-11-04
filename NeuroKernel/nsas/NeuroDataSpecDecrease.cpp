#include "stdafx.h"
#include "NeuroDataSpecDecrease.h"

#include "NeuroStorageAllocationSystem.h"

#include "NeuroDataAccessManager.h"
#include <algorithm>

using namespace np;
using namespace np::device;
using namespace np::nsas;

NeuroDataSpecDecrease::NeuroDataSpecDecrease(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec)
: NeuroDataSpecModify(nsas, allocSpec)
{
}

NeuroDataSpecDecrease::~NeuroDataSpecDecrease()
{
}

bool NeuroDataSpecDecrease::DeleteAll()
{
	return Decrease(0);
}

inline bool NeuroDataSpecDecrease::Decrease(neuro_u64 new_size)
{
	if (new_size >= m_allocSpec.size)
		return false;

	DEBUG_OUTPUT(L"old size[%llu], new size[%llu]", m_allocSpec.size, new_size);

	Pointer_Table_Spec_Path pt_spec_path;
	NeuroDataSpecTreeSearch nds_tree(m_nsas, m_allocSpec);
	if (!nds_tree.CreatePath(new_size, pt_spec_path))
	{
		DEBUG_OUTPUT(L"failed create path");
		return false;
	}
	NeuroDataSpecTreeSearch::_NDA_TREE_INFO new_nda_tree_info;
	nds_tree.GetDepthInfo(new_size, new_nda_tree_info);
	const neuro_32 new_root_index = pt_spec_path.total_depth - new_nda_tree_info.depth;

	// ���� pointer table path�� ������ index �������� �����ϸ� �ȴ�.
	neuro_32 sub_depth = pt_spec_path.total_depth;

	bool found_new_root = false;

	neuro_block total_deleted = 0;

	Pointer_Table_Spec_Base* cur_pt_spec = pt_spec_path.root;
	for(neuro_32 i=0; cur_pt_spec!=NULL; i++, sub_depth--)
	{
		if (sub_depth < 0)
		{
			DEBUG_OUTPUT(L"strange. sub_depth is minus");
			return false;
		}
		// ���� ������ ����Ʈ�� ����� ������ ������ ũ�⺸�� �۴ٸ� �������� �����ؾ� �Ѵ�.
		if (cur_pt_spec->search_node_scope < cur_pt_spec->GetPointerNodeCount())
		{
			// ���ο� ũ�⸦ �����ϴ� ��� �������� ó���ؾ� �ϱ� ������ cur_pt_spec->search_node_scope�� �״�� ��
			const neuro_u64 deleted = DeleteChildBlocks(sub_depth, *cur_pt_spec, cur_pt_spec->search_node_scope);
			if (deleted==neuro_last64)
			{
				DEBUG_OUTPUT(L"failed delete last pointer tables. total depth[%u], cur sub_depth[%u]"
					, pt_spec_path.total_depth, sub_depth);
				return false;
			}
			total_deleted += deleted;
		}
		cur_pt_spec->SetPointerNodeCount(cur_pt_spec->search_node_scope);

		// root ���� �ʿ���� ������ ������ �ĳ��鼭 �������ٰ�
		// �����ʰ͸� �ĳ±� ������ �θ��� pointer table�� list�� �ٷ� root list�� �ȴ�.
		if (i == new_root_index && cur_pt_spec != pt_spec_path.root)
		{
			if (cur_pt_spec->GetPointerNodeCount() > g_alloc_btree_root_count)
			{
				DEBUG_OUTPUT(L"new root is %th pointer table. but the size[%u] is bigger than %u"
					, cur_pt_spec->GetPointerNodeCount(), neuro_u32(g_alloc_btree_root_count));
				return false;
			}
			m_allocSpec.node_count = cur_pt_spec->GetPointerNodeCount();
			memcpy(m_allocSpec.block_bptree_root, cur_pt_spec->GetPointerNodeList(), m_allocSpec.node_count * sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));
		}

		cur_pt_spec = cur_pt_spec->child;
	}

	DEBUG_OUTPUT(L"total deleted count[%llu]", total_deleted);
	m_allocSpec.size = new_size;

	// ������ �غ���!!
	if (!Verify(new_nda_tree_info.depth))
	{
		DEBUG_OUTPUT(L"failed verify");
		return false;
	}

	DEBUG_OUTPUT(L"completed");
	return true;
}

inline neuro_block NeuroDataSpecDecrease::DeleteChildBlocks(neuro_u32 sub_depth, Pointer_Table_Spec_Base& pt, neuro_block start)
{
	if (start >= pt.GetPointerNodeCount())
	{
		DEBUG_OUTPUT(L"no child");
		return neuro_last64;
	}

	neuro_block total_deleted = 0;

	_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* node_list = pt.GetPointerNodeList();
	if (sub_depth > 0)	// sub_depth�� 0�̸� pointer node list�� data block�� ����Ų��.
	{
		const neuro_u32 child_depth = sub_depth - 1;

		Pointer_Table_Spec child_pt(NULL);
		for (neuro_block i = start, n = pt.GetPointerNodeCount(); i < n; i++)
		{
			if (!child_pt.ReadTable(m_nsas, node_list[i].block_no))
			{
				DEBUG_OUTPUT(L"failed to read pointer table");
				return neuro_last64;
			}

			neuro_block deleted_child = DeleteChildBlocks(child_depth, child_pt, 0);
			if (deleted_child==neuro_last64)
				return neuro_last64;

			total_deleted += deleted_child;
		}
	}
	const neuro_u32 del_count = pt.GetPointerNodeCount() - start;
	DEBUG_OUTPUT(L"dealloc blocks %u", del_count);
	if (m_nsas.DeallocBlocks(node_list + start, del_count) != del_count)
	{
		DEBUG_OUTPUT(L"failed to delete %u blocks", del_count);
		return neuro_last64;
	}

	return total_deleted + del_count;
}
