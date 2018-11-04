
#include "stdafx.h"
#include "NeuroStorageAllocationSystem.h"

#include "NeuroDataAccessManager.h"

#include <time.h>
#include <math.h>

using namespace np;
using namespace np::nsas;
using namespace np::device;

NeuroStorageAllocationSystem::NeuroStorageAllocationSystem(DeviceAdaptor& device)
: m_device(device)
{
	memset(&m_sbInfo, 0, sizeof(_NEURO_SUPER_BLOCK_INFO));
}

NeuroStorageAllocationSystem::~NeuroStorageAllocationSystem(void)
{
	ClearSuperBlockInfo();
}

// block ������ read/write
bool NeuroStorageAllocationSystem::BlockIO(neuro_block block, void* pBuffer, bool bWrite)
{
	neuro_u64 pos = block*m_sbInfo.nBlockSize;
	m_device.SetPosition(pos);

	neuro_u32 processed;
	if (bWrite)
		processed = m_device.Write(pBuffer, m_sbInfo.nBlockSize);
	else
		processed = m_device.Read(pBuffer, m_sbInfo.nBlockSize);

	if (processed != m_sbInfo.nBlockSize)
	{
		DEBUG_OUTPUT(L"failed %s. block=%u, position=%u, block size[%u], processed[%u]", bWrite ? L"write" : L"read",
			block, pos, m_sbInfo.nBlockSize, processed);
		return false;
	}
	return true;
}

/*
	��ü ũ�⿡ ���� block size�� �ʹ� ������ group descriptor�� �� ũ�Ⱑ block group�� ũ���� ���κ��� �����ϰų� �Ѿ �� �ִ�.
	����, ȿ������ ������ ���� group descriptor�� �����ϴ� block�� ���� �ִ� 64���� �Ͽ� �ּ����� block size�� ����Ѵ�.
*/
neuro_u32 NeuroStorageAllocationSystem::CalculateMinimumBlockSize(neuro_u64 nn_size)
{
	neuro_u32 max_desc_block_count=64;

/*
	nejuro_u32 block_count=NP_Util::CalculateCountPer(nn_size, block_size);
	neuro_u32 blocks_per_group=block_size*8;
	group_count= NP_Util::CalculateCountPer(block_count, blocks_per_group);

	neuro_u32 desc_size=sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR);
	desc_count_per_block = block_size/desc_size;

	max_desc_block_count=NP_Util::CalculateCountPer(group_count, desc_count_per_block);

	max_desc_block_count=(block_count/blocks_per_group)/(block_size/desc_size);
	max_desc_block_count=(block_count*desc_size)/(blocks_per_group*block_size);
	max_desc_block_count=(nn_size/block_size)*desc_size/(block_size*8*block_size);
	max_desc_block_count=(nn_size*desc_size)/(block_size*block_size*block_size*8);
	block_size*block_size*block_size*8=nn_size*desc_size/max_desc_block_count;
	block_size*block_size*block_size=nn_size*desc_size/(max_desc_block_count*8);
*/	
	double base = (nn_size / double(max_desc_block_count * 8)) * sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR);
	double d_min_block_size = pow(base, (double)(1.0 / 3.0));
	neuro_u32 log_base=d_min_block_size/1024;
	neuro_u32 log_block_size = log_base>0 ? log((double)log_base) : 0;
	neuro_u32 block_size=1024 << log_block_size;	// block ũ��� 1024������ �αװ����� ���ȴ�.

	return block_size;
}

inline neuro_u32 NeuroStorageAllocationSystem::CalculateLogblockSize(neuro_u32 block_size)
{
	double log_block_size = block_size>0 ? log((double)(block_size / 1024)) : 1;
	log_block_size = log_block_size / log(2.0);	// 1bit shift �� 2048�� �����̹Ƿ� log(2.0)���� �����ش�.
	return neuro_u32(log_block_size);
}

neuro_u32 NeuroStorageAllocationSystem::GetGroupSize(neuro_u32 block_size)
{
	// 1 bitmap can express [8 * block_size] blocks. 1 block's size is block_size
	// so size of group is [8*block_size] * block_size
	return 8 * block_size * block_size;
}

neuro_u32 NeuroStorageAllocationSystem::GetReservedBlocks(neuro_u32 group_desc_blocks_per_supergroup)
{
	neuro_u32 reserved = 1;	// nsas root root
	++reserved;	//_NEURAL_NETWORK_DEFINITION
	reserved += group_desc_blocks_per_supergroup;	// group description table
	++reserved;	// block bitmap

	return reserved;
}

bool NeuroStorageAllocationSystem::InitNSAS(neuro_u32 block_size)
{
	ClearSuperBlockInfo();

	nsas::_NEURO_STORAGE_ALLOC_SYSTEM_ROOT& nsas_root = m_sbInfo.nsas_root;

	nsas_root.SIGNATURE = mark_j_neuro;
	memcpy(nsas_root.version, last_neuro_version, sizeof(nsas_root.version));

	nsas_root.log_block_size = CalculateLogblockSize(block_size);

	/*	�ϳ��� group�� blockũ�⸸ŭ�� block���̺� ���� bitmap�� ������.
	�ϳ��� byte�� 8bit�� �����Ǿ� �ֱ� ������ block ũ�� * 8bit ��ŭ�� ���� ǥ���� �� �ִ�.
	*/
	block_size = 1024 << nsas_root.log_block_size;
	nsas_root.blocks_per_group = block_size * 8;

	if (m_device.IsFixedDevice())
	{
		const neuro_u32 max_group = m_device.GetMaxExtensibleSize() / (nsas_root.blocks_per_group * block_size);
		const neuro_u32 group_descs_per_block = block_size / sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR);
		nsas_root.group_desc_blocks_per_supergroup = max_group / group_descs_per_block;
	}
	else
		nsas_root.group_desc_blocks_per_supergroup = 1;	// ������ų�� ������ ��� super_group_count�� ������Ű���� �Ѵ�.

	nsas_root.super_group_count = 1;

	// calculate reserved blocks of group
	nsas_root.reserved_blocks_in_group = GetReservedBlocks(nsas_root.group_desc_blocks_per_supergroup);

	nsas_root.first_block_no = 0;
	nsas_root.nn_def_block_no = 1;

	SetNAS_SB(nsas_root.log_block_size);

	// ù group�� ���� ���ǰ� �Ǿ�� �Ѵ�.
	if (!AddGroup())
	{
		DEBUG_OUTPUT(L"failed add first group");
		return false;
	}

	return true;
}

bool NeuroStorageAllocationSystem::LoadNSAS()
{
	//Super Block�� ���� ����ü�� Supber Block�� �̿��Ͽ� ���� ���̴� �� ���� 
	ClearSuperBlockInfo();

	m_device.SetPosition(0);
	if (m_device.Read((neuro_u8*)&m_sbInfo.nsas_root, sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT)) != sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT))
	{
		memset(&m_sbInfo, 0, sizeof(_NEURO_SUPER_BLOCK_INFO));
		DEBUG_OUTPUT(L"failed load super block");
		return false;
	}

	SetNAS_SB(m_sbInfo.nsas_root.log_block_size);

	m_device.SetPosition(m_sbInfo.nsas_root.nn_def_block_no * m_sbInfo.nBlockSize);
	if (m_device.Read((neuro_u8*)&m_sbInfo.nn_def, sizeof(_NEURAL_NETWORK_DEFINITION)) != sizeof(_NEURAL_NETWORK_DEFINITION))
	{
		memset(&m_sbInfo, 0, sizeof(_NEURO_SUPER_BLOCK_INFO));
		DEBUG_OUTPUT(L"failed load network definition");
		return false;
	}

	if (!ReadWriteGroupDescs(false))
	{
		DEBUG_OUTPUT(L"failed read group descriptions");
		return false;
	}

	DEBUG_OUTPUT(L"blocks=%u, free=%u", m_sbInfo.nsas_root.blocks_count, m_sbInfo.nsas_root.free_blocks_count);
	return true;
}

void NeuroStorageAllocationSystem::SetNAS_SB(neuro_u32 log_block_size)
{
	m_sbInfo.nBlockSize = 1024 << log_block_size;

	m_sbInfo.nU32PerBlock = m_sbInfo.nBlockSize / sizeof(neuro_u32);

	m_sbInfo.group_descs_per_block = m_sbInfo.nBlockSize / sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR);
	m_sbInfo.groups_per_supergroup = m_sbInfo.nsas_root.group_desc_blocks_per_supergroup * m_sbInfo.group_descs_per_block;
	m_sbInfo.blocks_per_supergroup = m_sbInfo.groups_per_supergroup * m_sbInfo.nsas_root.blocks_per_group;
	m_sbInfo.max_group = m_sbInfo.nsas_root.super_group_count * m_sbInfo.groups_per_supergroup;
	m_sbInfo.max_size = (neuro_u64)m_sbInfo.max_group * (neuro_u64)m_sbInfo.nsas_root.blocks_per_group * (neuro_u64)m_sbInfo.nBlockSize;

	if (m_sbInfo.group_desc_array == NULL)
		m_sbInfo.group_desc_array = (_NEURO_ALLOC_GROUP_DESCRIPTOR*)malloc(m_sbInfo.max_group * sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR));
	else
		m_sbInfo.group_desc_array = (_NEURO_ALLOC_GROUP_DESCRIPTOR*)realloc(m_sbInfo.group_desc_array, m_sbInfo.max_group * sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR));

	m_sbInfo.group_start_block = m_sbInfo.nsas_root.nn_def_block_no + 1;
}

//  �ϴ� ù��° group���� writing�ߴµ� �����δ� block_group_number�� group���� �ؾ��Ѵ�.
bool NeuroStorageAllocationSystem::UpdateRootInfo()
{
	DEBUG_OUTPUT(L"blocks=%u, free=%u", m_sbInfo.nsas_root.blocks_count, m_sbInfo.nsas_root.free_blocks_count);

	m_device.SetPosition(0);
	neuro_u32 write = m_device.Write((neuro_u8*)&m_sbInfo.nsas_root, sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT));
	if (write != sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT))
	{
		DEBUG_OUTPUT(L"failed write root block[size:%u]. written[%u]", sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT), write);
		return false;
	}
	
	m_device.SetPosition(m_sbInfo.nsas_root.nn_def_block_no * m_sbInfo.nBlockSize);
	write = m_device.Write((neuro_u8*)&m_sbInfo.nn_def, sizeof(_NEURAL_NETWORK_DEFINITION));
	if (write != sizeof(_NEURAL_NETWORK_DEFINITION))
	{
		DEBUG_OUTPUT(L"failed write network definition block[size:%u]. written[%u]", sizeof(_NEURAL_NETWORK_DEFINITION), write);
		return false;
	}
	if (!m_device.Flush())
	{
		DEBUG_OUTPUT(L"failed device flush");
	}
	return true;
}

// Group Description Table �� �о����. �ϴ� ù��° group���� writing�ߴµ� �����δ� block_group_number�� group���� �ؾ��Ѵ�.
// root header�� group description table�� backup �� journaling �ý����� �Ϻ�Ǿ��� �� ����. �ٵ�, ���� file device������ �ʿ��ұ�?
bool NeuroStorageAllocationSystem::ReadWriteGroupDescs(bool isWrite)
{
	if (m_sbInfo.max_group != _msize(m_sbInfo.group_desc_array) / sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR))
	{
		DEBUG_OUTPUT(L"the size of group desc array is not match with max group");
		return false;
	}

	neuro_u32 read_group_desc = 0;

	neuro_block start_block = m_sbInfo.group_start_block;
	for (neuro_u32 super = 0; super < m_sbInfo.nsas_root.super_group_count; super++, start_block += m_sbInfo.blocks_per_supergroup)
	{
		for (neuro_u32 block = start_block; ; block++)
		{
			if (!BlockIO(block, m_sbInfo.group_desc_array + read_group_desc, isWrite))
			{
				if(!isWrite)
					memset(m_sbInfo.group_desc_array + read_group_desc, 0, m_sbInfo.nBlockSize);
				DEBUG_OUTPUT(L"failed %s group descriptors[%llu]", isWrite ? L"write" : L"read", block);
				return false;
			}
			read_group_desc += m_sbInfo.group_descs_per_block;

			if (read_group_desc >= m_sbInfo.nsas_root.group_count)	// ���� ���Ǵ� group ������ŭ�� ó���Ѵ�.
			{
				if (isWrite)
				{
					if (!m_device.Flush())
						DEBUG_OUTPUT(L"failed device flush");
				}

				return true;
			}
		}
	}
	DEBUG_OUTPUT(L"loading group descriptions is not completed");
	return false;
}

inline neuro_block NeuroStorageAllocationSystem::GetBlockNoFromGroupNo(neuro_u32 group_no, neuro_u32 block_index)
{
	return group_no*m_sbInfo.nsas_root.blocks_per_group + block_index;
}

// NAS�� block group ������ �����Ѵ�.
bool NeuroStorageAllocationSystem::AddGroup(neuro_u32 add_group_count)
{
	const neuro_u32 new_group_count = m_sbInfo.nsas_root.group_count + add_group_count;
	if (new_group_count > m_sbInfo.max_group)
	{
		if (m_device.IsFixedDevice())
		{
			DEBUG_OUTPUT(L"The count of existing group[%]s and addtional group[%u] is over than max group[%u]", m_sbInfo.nsas_root.group_count, add_group_count, m_sbInfo.max_group);
			return false;
		}
		// �̰� ���� �׽�Ʈ �ȵǾ���. �Ʒ� return false�� ���� �׽�Ʈ �غ�����.
		DEBUG_OUTPUT(L"adding new super group is not tested.");
		return false;
		m_sbInfo.nsas_root.super_group_count = NP_Util::CalculateCountPer(new_group_count, m_sbInfo.groups_per_supergroup);
		SetNAS_SB(m_sbInfo.nsas_root.log_block_size);
	}

	const neuro_u16 reserved_blocks_in_group = m_sbInfo.nsas_root.reserved_blocks_in_group;
	const neuro_u32 free_block_per_group = m_sbInfo.nsas_root.blocks_per_group - m_sbInfo.nsas_root.reserved_blocks_in_group;

	neuro_u32* pBitmap = (neuro_u32*)malloc(m_sbInfo.nBlockSize);
	neuro_u32 group_no = m_sbInfo.nsas_root.group_count;
	for(; group_no<new_group_count; group_no++)
	{
		{	// bitmap setting
			// reserved�� 32�� �̸��̴ϱ� �ϳ��� 32bit ������ �ص� ����
			neuro_u32 reserved_bits = (1 << reserved_blocks_in_group) - 1;
	#ifdef _DEBUG
			neuro_u32 temp = 0;
			for (neuro_u32 bit = 0; bit < reserved_blocks_in_group; bit++)
				temp |= (1 << bit);
			if (reserved_bits != temp)
			{
				DEBUG_OUTPUT(L"reserved bits is invalid");
				reserved_bits = temp;
			}
	#endif
			pBitmap[0] = reserved_bits;

			memset(pBitmap + 1, 0, m_sbInfo.nBlockSize - sizeof(neuro_u32));
		}

		// group descriptor
		_NEURO_ALLOC_GROUP_DESCRIPTOR& desc = m_sbInfo.group_desc_array[group_no];
		memset(&desc, 0, sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR));
		desc.bitmap_block_index = reserved_blocks_in_group - 1;	// ù��° group�� bitmap
		desc.free_blocks_count = free_block_per_group;

		neuro_block bitmap_block_no = GetBlockNoFromGroupNo(group_no, desc.bitmap_block_index);
		if (!BlockIO(bitmap_block_no, (neuro_u8*)pBitmap, true))
		{
			DEBUG_OUTPUT(L"failed write bitmap[%llu block] for group[%u]", bitmap_block_no, group_no);
			break;
		}

		DEBUG_OUTPUT(L"add group %u", group_no);
	}
	free(pBitmap);

	if (group_no < new_group_count)
		return false;

	nsas::_NEURO_STORAGE_ALLOC_SYSTEM_ROOT old_root;
	memcpy(&old_root, &m_sbInfo.nsas_root, sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT));

	m_sbInfo.nsas_root.group_count = new_group_count;
	m_sbInfo.nsas_root.blocks_count = m_sbInfo.nsas_root.group_count*m_sbInfo.nsas_root.blocks_per_group;
	m_sbInfo.nsas_root.free_blocks_count += add_group_count * free_block_per_group;

	// �̶� ù��° group�� ���� update ����� �ϳ�. �׷� UpdateRootInfo�� �������� �ε�...
	m_sbInfo.nsas_root.block_group_number = m_sbInfo.nsas_root.group_count - 1;

	if (!ReadWriteGroupDescs(true))
	{
		DEBUG_OUTPUT(L"failed write group descriptions");
		memcpy(&m_sbInfo.nsas_root, &old_root, sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT));
		return false;
	}
	if (!UpdateRootInfo())
	{
		DEBUG_OUTPUT(L"failed write super block");
		memcpy(&m_sbInfo.nsas_root, &old_root, sizeof(_NEURO_STORAGE_ALLOC_SYSTEM_ROOT));
		return false;
	}
	return true;
}

//�����͸� ������ ����� �Ҵ��ϰ�, Bitmap�� ������Ʈ 
neuro_u32 NeuroStorageAllocationSystem::AllocBlocks(neuro_block relativeBlock, neuro_u32 nAllocBlock, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE *pPNodeList)
{ 
	if (nAllocBlock > m_sbInfo.nsas_root.free_blocks_count)
	{
		neuro_u32 group_count = NP_Util::CalculateCountPer(nAllocBlock - m_sbInfo.nsas_root.free_blocks_count, m_sbInfo.nsas_root.blocks_per_group);
		if (AddGroup(group_count) == neuro_last32)	// group ������ �ø��� file device������ �ʹ� Ŀ����. ������, ���� ���� ��ϸ�ŭ�� writing�ϴϱ� ������!
		{
			DEBUG_OUTPUT(L"failed to add group");
			return neuro_last32;
		}
	}

	neuro_u32 relative_group = GetGroupFromBlock(relativeBlock);

	neuro_u8* pBlockBitmapBuffer = (neuro_u8*)malloc(m_sbInfo.nBlockSize);

	neuro_u32 nTotalAlloc = 0;
	nTotalAlloc += AllocBlockInGroup(pBlockBitmapBuffer, relative_group, relative_group+1, nAllocBlock - nTotalAlloc, pPNodeList + nTotalAlloc);
	nTotalAlloc += AllocBlockInGroup(pBlockBitmapBuffer, 0, relative_group, nAllocBlock - nTotalAlloc, pPNodeList + nTotalAlloc);
	nTotalAlloc += AllocBlockInGroup(pBlockBitmapBuffer, relative_group + 1, m_sbInfo.nsas_root.group_count, nAllocBlock - nTotalAlloc, pPNodeList + nTotalAlloc);

	free(pBlockBitmapBuffer);

	if (!ReadWriteGroupDescs(true))	// ����� group�� �����ϵ��� �����ϸ� AllocBlockInGroup ���� �Űܾ� �Ѵ�.
	{
		DEBUG_OUTPUT(L"failed write group descriptions");
		return 0;
	}

	if (!UpdateRootInfo())
	{
		DEBUG_OUTPUT(L"failed sync super glock");
		return 0;
	}

	if (nTotalAlloc<nAllocBlock)
	{
		DEBUG_OUTPUT(L"total alloc(%u) < alloc(%u)", nTotalAlloc, nAllocBlock);
	}
	return nTotalAlloc;
}

neuro_u32 NeuroStorageAllocationSystem::AllocBlockInGroup(neuro_u8* pBlockBitmapBuffer, neuro_u32 group_start, neuro_u32 group_last, neuro_u32 nAllocBlock, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE *pPNodeList)
{
	neuro_u32* p32Bitmap = (neuro_u32*)pBlockBitmapBuffer;

	neuro_u32 nTotalAlloc=0;

	if (&m_sbInfo.nsas_root.free_blocks_count == 0)
	{
		DEBUG_OUTPUT(L"no free blocks");
		return 0;
	}

	for (neuro_u32 group_no = group_start; group_no<group_last && nTotalAlloc < nAllocBlock; group_no++)
	{
		_NEURO_ALLOC_GROUP_DESCRIPTOR& desc = m_sbInfo.group_desc_array[group_no];

		if (desc.free_blocks_count == 0)
		{
			DEBUG_OUTPUT(L"no free blocks. group[%u]", group_no);
			continue;
		}

		neuro_block bitmap_block_no = GetBlockNoFromGroupNo(group_no, desc.bitmap_block_index);
		BlockIO(bitmap_block_no, p32Bitmap, false);

		neuro_u32 first_block_no = GetBlockNoFromGroupNo(group_no, 0);

		// block�� ũ�⸦ 4byte(neuro_u32 ũ��)�� ����� ���. ��. little endian. ���߿� Sun? �� ����. �׶�����
		for(int i = 0;i < m_sbInfo.nU32PerBlock && nTotalAlloc < nAllocBlock;i++)
		{ 
			if (p32Bitmap[i] == 0xFFFFFFFF)
				continue;

			neuro_u32 value = p32Bitmap[i];
			for (int bit = 0; bit<32 && nTotalAlloc < nAllocBlock; bit++)
			{ 
				neuro_u32 shift = 1 << bit;
				if ((shift & value) != 0)
					continue;

				value |= shift;// block �Ҵ�

				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE& pnode=pPNodeList[nTotalAlloc++];
				pnode.block_no = first_block_no + (i * 32 + bit); // �Ҵ�� block ��ȣ ����

//				DEBUG_OUTPUT(L"block[%llu] is allocated", pnode.block_no);
			}
			p32Bitmap[i] = value;
		}

		BlockIO(bitmap_block_no, p32Bitmap, true);

		desc.free_blocks_count -= nTotalAlloc;		// descriptor�� ����Ǿ��� ������ �����ؾ��Ѵ�.
	}

	m_sbInfo.nsas_root.free_blocks_count -= nTotalAlloc;

	return nTotalAlloc; 
}

inline neuro_u32 NeuroStorageAllocationSystem::GetGroupFromBlock(neuro_block block)
{
	return block/m_sbInfo.nsas_root.blocks_per_group;
}

neuro_u32 NeuroStorageAllocationSystem::DeallocBlocks(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* pPNodeList, neuro_u32 nBlock)
{
	if (nBlock == 0)
		return 0;

	neuro_u8* pBlockBitmapBuffer=(neuro_u8*)malloc(m_sbInfo.nBlockSize);
	memset(pBlockBitmapBuffer, 0, m_sbInfo.nBlockSize);

	neuro_u32 prev_group_no = neuro_last32;

	bool bRet = true;

	neuro_u32 delete_block=0;
	for(;delete_block<nBlock; delete_block++)
	{
		neuro_block block_no = pPNodeList[delete_block].block_no;
		if(block_no == 0)
		{
			DEBUG_OUTPUT(L"failed : block no in block pointer node list[%u] is zero", delete_block);
			bRet = false;
			break;
		}
		if (block_no >= m_sbInfo.nsas_root.blocks_count)
		{
			DEBUG_OUTPUT(L"failed : block no[%llu] in block pointer node list[%u] is over block count[%llu]", block_no, delete_block, m_sbInfo.nsas_root.blocks_count);
			bRet = false;
			break;
		}
#ifdef _DEBUG
		if (block_no == 4374045682979117159)
			int a = 0;
#endif

		neuro_u32 group_no=GetGroupFromBlock(block_no);	// block�� ���� ���� group�� ã�Ƽ� block bitmap�� ���ؾ��Ѵ�.
		_NEURO_ALLOC_GROUP_DESCRIPTOR& desc = m_sbInfo.group_desc_array[group_no];
		if (group_no != prev_group_no)	// group�� �ٲ������ ���� ����!
		{
			if (prev_group_no != neuro_last32)	// ���� block bitmap�� �����Ѵ�.
			{
				neuro_block bitmap_block_no = GetBlockNoFromGroupNo(prev_group_no, m_sbInfo.group_desc_array[prev_group_no].bitmap_block_index);
				BlockIO(bitmap_block_no, pBlockBitmapBuffer, true);	// ���� �׷��� bitmap block�� �����Ѵ�.
			}
			neuro_block bitmap_block_no = GetBlockNoFromGroupNo(group_no, desc.bitmap_block_index);
			BlockIO(bitmap_block_no, pBlockBitmapBuffer, false);
			prev_group_no = group_no;
		}

		neuro_u32 bpos = block_no - (group_no * (neuro_u32)m_sbInfo.nsas_root.blocks_per_group);

		neuro_u32 bytes = bpos / 8;
		neuro_u32 bits = bpos % 8; 

		pBlockBitmapBuffer[bytes] &= ~(1 <<bits); 

		++desc.free_blocks_count;// descriptor�� ����Ǿ��� ������ �����ؾ��Ѵ�.	// Super block�� ���??
		++m_sbInfo.nsas_root.free_blocks_count;
	}

	if (bRet && prev_group_no == neuro_last32)
	{
		DEBUG_OUTPUT(L"some strange. prev group no is invalid");
		bRet = false;
	}
	if(bRet)
	{
		neuro_block bitmap_block_no = GetBlockNoFromGroupNo(prev_group_no, m_sbInfo.group_desc_array[prev_group_no].bitmap_block_index);
		if (!BlockIO(bitmap_block_no, pBlockBitmapBuffer, true))
		{
			DEBUG_OUTPUT(L"failed write bitmap block[%u], group[%u]", bitmap_block_no, prev_group_no);
			bRet = false;
		}
	}

	free(pBlockBitmapBuffer);

	if (!bRet)
		return 0;

	if (!ReadWriteGroupDescs(true))
	{
		DEBUG_OUTPUT(L"failed write super group");
		return 0;
	}
	if (!UpdateRootInfo())
	{
		DEBUG_OUTPUT(L"failed write super block");
		return 0;
	}

	return delete_block;
}
