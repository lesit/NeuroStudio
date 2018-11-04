
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

// block 단위의 read/write
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
	전체 크기에 비해 block size가 너무 작으면 group descriptor의 총 크기가 block group의 크기의 상당부분을 차지하거나 넘어설 수 있다.
	따라서, 효율적인 관리를 위해 group descriptor가 차지하는 block의 수를 최대 64개로 하여 최소한의 block size를 계산한다.
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
	neuro_u32 block_size=1024 << log_block_size;	// block 크기는 1024단위의 로그값으로 계산된다.

	return block_size;
}

inline neuro_u32 NeuroStorageAllocationSystem::CalculateLogblockSize(neuro_u32 block_size)
{
	double log_block_size = block_size>0 ? log((double)(block_size / 1024)) : 1;
	log_block_size = log_block_size / log(2.0);	// 1bit shift 당 2048씩 움직이므로 log(2.0)으로 나눠준다.
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

	/*	하나의 group은 block크기만큼의 block테이블에 대한 bitmap을 가진다.
	하나의 byte가 8bit로 구성되어 있기 때문에 block 크기 * 8bit 만큼의 블럭을 표현할 수 있다.
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
		nsas_root.group_desc_blocks_per_supergroup = 1;	// 고정시킬수 없으며 대신 super_group_count를 증가시키도록 한다.

	nsas_root.super_group_count = 1;

	// calculate reserved blocks of group
	nsas_root.reserved_blocks_in_group = GetReservedBlocks(nsas_root.group_desc_blocks_per_supergroup);

	nsas_root.first_block_no = 0;
	nsas_root.nn_def_block_no = 1;

	SetNAS_SB(nsas_root.log_block_size);

	// 첫 group에 대해 정의가 되어야 한다.
	if (!AddGroup())
	{
		DEBUG_OUTPUT(L"failed add first group");
		return false;
	}

	return true;
}

bool NeuroStorageAllocationSystem::LoadNSAS()
{
	//Super Block의 보조 구조체로 Supber Block을 이용하여 자주 쓰이는 값 저장 
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

//  일단 첫번째 group에만 writing했는데 실제로는 block_group_number의 group에도 해야한다.
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

// Group Description Table 을 읽어들임. 일단 첫번째 group에만 writing했는데 실제로는 block_group_number의 group에도 해야한다.
// root header와 group description table의 backup 과 journaling 시스템이 완비되었을 때 하자. 근데, 굳이 file device에서는 필요할까?
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

			if (read_group_desc >= m_sbInfo.nsas_root.group_count)	// 현재 사용되는 group 개수만큼만 처리한다.
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

// NAS는 block group 단위로 증가한다.
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
		// 이건 아직 테스트 안되었다. 아래 return false를 빼고 테스트 해봐야함.
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
			// reserved가 32개 미만이니까 하나의 32bit 변수만 해도 가능
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
		desc.bitmap_block_index = reserved_blocks_in_group - 1;	// 첫번째 group의 bitmap
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

	// 이때 첫번째 group도 같이 update 해줘야 하나. 그럼 UpdateRootInfo도 마찬가지 인데...
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

//데이터를 저장할 블록을 할당하고, Bitmap을 업데이트 
neuro_u32 NeuroStorageAllocationSystem::AllocBlocks(neuro_block relativeBlock, neuro_u32 nAllocBlock, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE *pPNodeList)
{ 
	if (nAllocBlock > m_sbInfo.nsas_root.free_blocks_count)
	{
		neuro_u32 group_count = NP_Util::CalculateCountPer(nAllocBlock - m_sbInfo.nsas_root.free_blocks_count, m_sbInfo.nsas_root.blocks_per_group);
		if (AddGroup(group_count) == neuro_last32)	// group 단위로 늘리면 file device에서는 너무 커진다. 하지만, 실제 사용된 블록만큼만 writing하니까 괜찮다!
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

	if (!ReadWriteGroupDescs(true))	// 변경된 group만 저장하도록 구현하면 AllocBlockInGroup 으로 옮겨야 한다.
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

		// block의 크기를 4byte(neuro_u32 크기)로 나누어서 계산. 즉. little endian. 나중에 Sun? 은 몰라. 그때가서
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

				value |= shift;// block 할당

				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE& pnode=pPNodeList[nTotalAlloc++];
				pnode.block_no = first_block_no + (i * 32 + bit); // 할당된 block 번호 저장

//				DEBUG_OUTPUT(L"block[%llu] is allocated", pnode.block_no);
			}
			p32Bitmap[i] = value;
		}

		BlockIO(bitmap_block_no, p32Bitmap, true);

		desc.free_blocks_count -= nTotalAlloc;		// descriptor가 변경되었기 때문에 저장해야한다.
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

		neuro_u32 group_no=GetGroupFromBlock(block_no);	// block에 대한 실제 group을 찾아서 block bitmap을 구해야한다.
		_NEURO_ALLOC_GROUP_DESCRIPTOR& desc = m_sbInfo.group_desc_array[group_no];
		if (group_no != prev_group_no)	// group이 바뀌었으면 새로 읽자!
		{
			if (prev_group_no != neuro_last32)	// 이전 block bitmap을 저장한다.
			{
				neuro_block bitmap_block_no = GetBlockNoFromGroupNo(prev_group_no, m_sbInfo.group_desc_array[prev_group_no].bitmap_block_index);
				BlockIO(bitmap_block_no, pBlockBitmapBuffer, true);	// 이전 그룹의 bitmap block을 저장한다.
			}
			neuro_block bitmap_block_no = GetBlockNoFromGroupNo(group_no, desc.bitmap_block_index);
			BlockIO(bitmap_block_no, pBlockBitmapBuffer, false);
			prev_group_no = group_no;
		}

		neuro_u32 bpos = block_no - (group_no * (neuro_u32)m_sbInfo.nsas_root.blocks_per_group);

		neuro_u32 bytes = bpos / 8;
		neuro_u32 bits = bpos % 8; 

		pBlockBitmapBuffer[bytes] &= ~(1 <<bits); 

		++desc.free_blocks_count;// descriptor가 변경되었기 때문에 저장해야한다.	// Super block은 어떻게??
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
