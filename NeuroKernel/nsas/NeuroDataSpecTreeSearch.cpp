#include "stdafx.h"

#include "NeuroDataSpecTreeSearch.h"

#include "NeuroDataAccessManager.h"

using namespace np;
using namespace np::nsas;

NeuroDataSpecTreeSearch::NeuroDataSpecTreeSearch(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC &allocSpec)
: m_nsas(nsas), m_allocSpec(allocSpec), m_pointersPerBlock(GetPointersPerBlock(nsas.GetBlockSize()))
{
}

NeuroDataSpecTreeSearch::~NeuroDataSpecTreeSearch()
{
}

bool NeuroDataSpecTreeSearch::CreatePath(neuro_u64 size, Pointer_Table_Spec_Path& ret) const
{
	if (ret.leaf)
		delete ret.leaf;

	ret.root = new Root_Spec(m_allocSpec);
	ret.leaf = ret.root;
	if (size == 0)
	{
		DEBUG_OUTPUT(L"size is zero");
		return true;
	}

#ifdef _DEBUG
	if (m_allocSpec.size > 9000)
		int a = 0;

	if (ret.total_depth > 1)
		int a = 0;
#endif

	_NDA_TREE_INFO nda_tree_info;
	ret.total_depth = GetDepthInfo(m_allocSpec.size, nda_tree_info);

	neuro_u64 cur_level_node_size = size;

	neuro_u64 size_per_node = nda_tree_info.size_per_root_node;
	ret.leaf->search_node_scope = NP_Util::CalculateCountPer(cur_level_node_size, size_per_node);
	
	for (neuro_32 depth = 1; depth <= ret.total_depth; depth++)
	{
		if (ret.leaf->search_node_scope == 0)
		{
			DEBUG_OUTPUT(L"leaf search node scope is zero");
			return false;
		}
		neuro_block block_no = ret.leaf->GetPointerNodeList()[ret.leaf->search_node_scope - 1].block_no;

		Pointer_Table_Spec_Base* parent = ret.leaf;
		Pointer_Table_Spec* new_spec = new Pointer_Table_Spec(parent);
		ret.leaf = new_spec;

		if (!new_spec->ReadTable(m_nsas, block_no))
		{
			DEBUG_OUTPUT(L"failed to read block[%llu] of pointer_node[%u]", block_no, parent->search_node_scope);
			return false;
		}

		cur_level_node_size -= (parent->search_node_scope - 1) * size_per_node;	// 부모의 앞 형제에서 처리한건 빼준다.

		size_per_node /= m_pointersPerBlock;
		if (size_per_node == 0)
		{
			DEBUG_OUTPUT(L"strange status. depth=%d", neuro_32(depth));
			return false;
		}
		ret.leaf->search_node_scope = NP_Util::CalculateCountPer(cur_level_node_size, size_per_node);
	}

//	DEBUG_OUTPUT(L"depth is %u", neuro_u32(ret.total_depth));
	return true;
}

inline neuro_32 NeuroDataSpecTreeSearch::GetDepthInfo(neuro_u64 size, NeuroDataSpecTreeSearch::_NDA_TREE_INFO& ret) const
{
	neuro_32& depth = ret.depth;
	neuro_u64& size_per_root_node = ret.size_per_root_node;

	depth = 0;
	size_per_root_node = m_nsas.GetBlockSize();

	// 1 level : g_alloc_btree_root_count x block_size / sizeof(neuro_block) 개의 leaf block 정의 가능
	// 2 level : 위 최대 크기 / sizeof(neuro_block) * block_size 까지 가능

	while (size > size_per_root_node * g_alloc_btree_root_count)
	{
		size_per_root_node *= m_pointersPerBlock;
		++depth;
	}

#ifdef _DEBUG
	if (depth == 10)
		int a = 0;
#endif

	return depth;
}
