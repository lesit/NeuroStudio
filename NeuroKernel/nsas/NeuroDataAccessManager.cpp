#include "stdafx.h"
#include "NeuroDataAccessManager.h"

#include "NeuroStorageAllocationSystem.h"

#include "NeuroDataSpecIncrease.h"
#include "NeuroDataSpecDecrease.h"

using namespace np;
using namespace np::device;
using namespace np::nsas;

NeuroDataAccessManager::NeuroDataAccessManager(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec)
: NeuroDataAccessManager(nsas, allocSpec, false)
{
}

NeuroDataAccessManager::NeuroDataAccessManager(const NeuroStorageAllocationSystem& nsas, const _NEURO_DATA_ALLOC_SPEC& allocSpec)
: NeuroDataAccessManager(const_cast<NeuroStorageAllocationSystem&>(nsas), const_cast<_NEURO_DATA_ALLOC_SPEC&>(allocSpec), true)
{
}

NeuroDataAccessManager::NeuroDataAccessManager(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec, bool read_only)
: m_bReadOnly(read_only), m_nsas(nsas), m_allocSpec(allocSpec), m_cached_block(nsas.GetBlockSize())
{
	Initialize(m_nsas);

	SetDataPointer(0);
}

NeuroDataAccessManager::NeuroDataAccessManager(NeuroDataAccessManager& src)
: m_bReadOnly(src.m_bReadOnly), m_nsas(src.m_nsas), m_allocSpec(src.m_allocSpec), m_cached_block(src.m_nsas.GetBlockSize())
{
	Initialize(m_nsas);

	memcpy(m_leaf_table.block_data, src.m_leaf_table.block_data, m_nBlockSize);

	memcpy(&m_data_block_info, &src.m_data_block_info, sizeof(_DATA_BLOCK_INFO));

	m_position = src.m_position;
}

NeuroDataAccessManager::~NeuroDataAccessManager()
{
	if (m_leaf_table.block_data)
		free(m_leaf_table.block_data);
}

void NeuroDataAccessManager::Initialize(NeuroStorageAllocationSystem& nsas)
{
	m_nBlockSize = nsas.GetBlockSize();

	m_position = 0;

	memset(&m_leaf_table, 0, sizeof(m_leaf_table));
	m_leaf_table.block_data = (neuro_u8*)calloc(m_nBlockSize, sizeof(neuro_u8));

	memset(&m_data_block_info, 0, sizeof(m_data_block_info));
}

bool NeuroDataAccessManager::SetDataPointer(neuro_u64 pos)
{
	memset(&m_data_block_info, 0, sizeof(m_data_block_info));

	if (pos == neuro_last64)
	{
		pos = m_allocSpec.size;
	}
	else if (pos > m_allocSpec.size)
	{
		DEBUG_OUTPUT(L"position[%llu] is over size[%llu]", pos, m_allocSpec.size);
		return false;
	}

	m_position = pos;

	if (m_allocSpec.size == 0)
		return true;

#ifdef _DEBUG
	if (pos == 2000)
		int a = 0;
#endif

	Pointer_Table_Spec_Path pt_spec_path;
	NeuroDataSpecTreeSearch nds_tree(m_nsas, m_allocSpec);
	if (!nds_tree.CreatePath(m_position+1, pt_spec_path))
	{
		DEBUG_OUTPUT(L"failed create path");
		return false;
	}
	if (pt_spec_path.leaf->search_node_scope == 0)
	{
		DEBUG_OUTPUT(L"strange! child scope is 0");
		return false;
	}
//	DEBUG_OUTPUT(L"depth[%d], cur pointer table[%llu]", pt_spec_path.total_depth, pt_spec_path.leaf->GetTableBlockNo());

	_POINTER_TABLE_HEADER* pt_header = m_leaf_table.GetHeader();
	pt_header->next_pointer_table = pt_spec_path.leaf->GetNextPointerTable();
	pt_header->node_count = pt_spec_path.leaf->GetPointerNodeCount();
	if (pt_header->node_count < pt_spec_path.leaf->search_node_scope)
	{
		DEBUG_OUTPUT(L"strange node count[%u] this must bigger than node scope[%u]!", pt_header->node_count, pt_spec_path.leaf->search_node_scope);
		return false;
	}

	_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* pointer_node = m_leaf_table.GetPointerNodeList();
	memcpy(pointer_node, pt_spec_path.leaf->GetPointerNodeList(), pt_header->node_count*sizeof(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE));

	if (!m_data_block_info.SetDataBlockInfo(m_nBlockSize, m_allocSpec.size, m_leaf_table, pt_spec_path.leaf->search_node_scope - 1, pos%m_nBlockSize))
	{
		DEBUG_OUTPUT(L"failed SetDataBlockInfo");
		return false;
	}
	return true;
}

bool NeuroDataAccessManager::ReadData(void* buffer, neuro_u64 size) const
{
	return const_cast<NeuroDataAccessManager*>(this)->ReadWriteData(false, buffer, size);
}

bool NeuroDataAccessManager::WriteData(const void* buffer, neuro_u64 size)
{
	return ReadWriteData(true, (void*) buffer, size);
}

inline bool NeuroDataAccessManager::ReadWriteData(bool bWrite, void* buffer, neuro_u64 size)
{ 
	if (size == 0)
		return true;

	const wchar_t* mode = bWrite ? L"write" : L"read";
	if (bWrite)
	{
		if (m_bReadOnly)
		{
			DEBUG_OUTPUT(L"%s mode. failed because of read only", mode);
			return false;
		}
	}

	if (m_position + size>m_allocSpec.size)
	{
		if (bWrite)
		{
			NeuroDataSpecIncrease alloc(m_nsas, m_allocSpec);
			if (!alloc.Increase(m_position + size))
			{
				DEBUG_OUTPUT(L"%s mode. failed increase", mode);
				return false;
			}

			// �ٲ� pointer table�� �����ϱ� ���ؼ�..
			if (!SetDataPointer(m_position))
			{
				DEBUG_OUTPUT(L"%s mode. failed SetDataPointer", mode);
				return false;
			}
		}
		else
		{
			DEBUG_OUTPUT(L"%s mode. the size is over. pos[%llu], size[%llu], alloc size[%llu", mode, m_position, size, m_allocSpec.size);
			return false;
		}
	}

	if (m_data_block_info.datablock_no == 0)
	{
		DEBUG_OUTPUT(L"%s mode. no current data block", mode);
		return false;
	}

	neuro_u64 readwrite = 0;
	while (true)
	{
		neuro_u32 count = min(size - readwrite, m_data_block_info.data_size - m_data_block_info.posInDataBlock);
		if (m_cached_block.datablock_no!=m_data_block_info.datablock_no
			&& (!bWrite || m_data_block_info.posInDataBlock > 0 || count < m_data_block_info.data_size))
		{
			// �ι�° ���ǹ���, cached ���� �ʾҴ� �ϴ���
			// write ��忡�� ������ ��� ��ü�� writing �� ��� ���� ���� �� ���� �ʿ䰡 �����Ƿ� ���ܽ�Ű�� ���� ����
			if (!m_nsas.BlockIO(m_data_block_info.datablock_no, m_cached_block.buffer, false))
			{
				DEBUG_OUTPUT(L"%s mode. read data block[%llu]", mode, m_data_block_info.datablock_no);
				return false;
			}
		}
		// �� ���ǹ��� ���Ե��� �ʾ�����, write ����̰� ��� ��ü�� writing �� ��찡 �����Ƿ�
		m_cached_block.datablock_no = m_data_block_info.datablock_no;

		if (bWrite)
		{
			memcpy(m_cached_block.buffer + m_data_block_info.posInDataBlock, buffer, count);
			if (!m_nsas.BlockIO(m_data_block_info.datablock_no, m_cached_block.buffer, true))
			{
				DEBUG_OUTPUT(L"%s mode. failed write block[%llu]", mode, m_data_block_info.datablock_no);
				return false;
			}
		}
		else
		{
			if (m_data_block_info.data_size<m_data_block_info.posInDataBlock)
			{
				DEBUG_OUTPUT(L"%s mode. the position[%u] is over data block[%u] size[%u]", mode
					, m_data_block_info.posInDataBlock, m_data_block_info.datablock_no, m_data_block_info.data_size);
				return false;
			}
			memcpy(buffer, m_cached_block.buffer + m_data_block_info.posInDataBlock, count);
		}
		m_data_block_info.posInDataBlock += count;

		readwrite += count;
		buffer = (neuro_u8*)buffer + count;

		if (readwrite == size)
		{
			// ���� ������ ����� �� ó�������� ���� ������� �̵��� �ش�.
			if (m_data_block_info.posInDataBlock == m_data_block_info.data_size)
			{
				if (!MoveNextBlock())
				{
					if (m_position + readwrite != m_allocSpec.size)
					{
						DEBUG_OUTPUT(L"%s mode. no more data block when preload. but must be have", mode);
					}
				}
			}
			break;
		}

		if (!MoveNextBlock())
		{
			DEBUG_OUTPUT(L"%s mode. no more data block. not completed. %llu / %llu", mode, readwrite, size);
			break;
		}
	}

	m_position += readwrite;
	return readwrite == size;
}

// �Ʒ�ó�� tree traverse �� �ʿ� ����. �����ͺ�� ������ ���̺�ȿ��� ��� �̵��ϴٰ�, ������ leaf���� ���� ������ ���̺� ������� �̵��ϸ� �ȴ�?
inline bool NeuroDataAccessManager::MoveNextBlock()
{
	_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* pointer_node = m_leaf_table.GetPointerNodeList();
	_POINTER_TABLE_HEADER* pt_header = m_leaf_table.GetHeader();

	neuro_u32 node_index = m_data_block_info.datablock_index + 1;	// ���� ���� ������ ��Ͽ� ���� index

	/* ���� data node index�� ���� ������ ���̺��� node �������� ũ�ٸ�, ���� ������ ���̺�� �Ѿ�� �Ѵ�.
	�̷��� �����ν�, data node index�� ������ ������ ������ ���̺��� �׻� �ֽ� ������ �����Ѵ�.
	*/

#ifdef _NDA_DEBUG
	DEBUG_OUTPUT(L"next index=%u, node count=%u", node_index, pt_header->node_count);
#endif

	memset(&m_data_block_info, 0, sizeof(_DATA_BLOCK_INFO));
	if (node_index >= pt_header->node_count)
	{
		neuro_block next_pt = pt_header->next_pointer_table;
		if (next_pt == 0)
		{
#ifdef _NDA_DEBUG
			DEBUG_OUTPUT(L"no next_table. might be eof");
#endif
			// ���̻� ������ ����� ���ٴ� ���� m_data_block_info.datablock_no == 0 ���� �� �� ����
			return false;
		}

		if (!m_nsas.BlockIO(next_pt, m_leaf_table.block_data, false))
		{
			DEBUG_OUTPUT(L"failed read next block[%llu]", next_pt);
			return false;
		}

		node_index = 0;

		DEBUG_OUTPUT(L"move next pointer table[%llu]", m_leaf_table.GetHeader()->next_pointer_table);
	}

	if (!m_data_block_info.SetDataBlockInfo(m_nBlockSize, m_allocSpec.size, m_leaf_table, node_index, 0))
	{
		DEBUG_OUTPUT(L"failed SetDataBlockInfo");
		return false;
	}
	return true;
}

inline bool _DATA_BLOCK_INFO::SetDataBlockInfo(neuro_u32 nBlockSize, neuro_u64 total_size, const _POINTER_TABLE_INFO& pt_info, neuro_u32 node_index, neuro_u32 _posInDataBlock)
{
	const _POINTER_TABLE_HEADER* pt_header = pt_info.GetHeader();
	const _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* pointer_node = pt_info.GetPointerNodeList();

	datablock_index = node_index;
	datablock_no = pointer_node[datablock_index].block_no;

	// ������ ������ ���̺��̸鼭 ������ ������ ����� ���
	if (pt_header->next_pointer_table == 0 && (node_index + 1) == pt_header->node_count)
	{
		data_size = total_size % nBlockSize;
		if (data_size == 0)	
		{
			data_size = nBlockSize;
			DEBUG_OUTPUT(L"total[%llu] % block[%u] is zero. so, set block size", total_size, nBlockSize);
		}
	}
	else
		data_size = nBlockSize;

#ifdef _DEBUG
	if (_posInDataBlock > 0)
		int a = 0;
#endif

	posInDataBlock = _posInDataBlock;

	return true;
}

bool NeuroDataAccessManager::SetSize(neuro_u64 size)
{
	if (m_bReadOnly)
	{
		DEBUG_OUTPUT(L"failed because of read only");
		return false;
	}

	if (size == m_allocSpec.size)
		return true;

	if(size<m_allocSpec.size)
	{
		NeuroDataSpecDecrease alloc(m_nsas, m_allocSpec);
		if (!alloc.Decrease(size))
			return false;
	}
	else
	{
		NeuroDataSpecIncrease alloc(m_nsas, m_allocSpec);
		if (!alloc.Increase(size))
			return false;
	}

	return SetDataPointer(0);
}

bool NeuroDataAccessManager::DeleteFromCurrent()
{
	if (m_bReadOnly)
	{
		DEBUG_OUTPUT(L"failed because of read only");
		return false;
	}

	NeuroDataSpecDecrease alloc(m_nsas, m_allocSpec);
	return alloc.Decrease(m_position);
}
