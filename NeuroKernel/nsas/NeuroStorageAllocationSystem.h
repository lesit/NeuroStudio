#if !defined(NEURO_ALLOCATION_SYSTEM_MANAGER_H)
#define NEURO_ALLOCATION_SYSTEM_MANAGER_H

#include "common.h"

#include "storage/DeviceAdaptor.h"

#include "NeuroStorageAllocationTableSpec.h"

namespace np
{
	namespace nsas
	{
		class NeuroStorageAllocationSystem
		{
		public:
			NeuroStorageAllocationSystem(device::DeviceAdaptor& device);
			virtual ~NeuroStorageAllocationSystem();

			static neuro_u32 CalculateMinimumBlockSize(neuro_u64 nn_size);
			static neuro_u32 CalculateLogblockSize(neuro_u32 block_size);

			bool InitNSAS(neuro_u32 block_size);
			bool LoadNSAS();
			void CompleteUpdate()	// 모든 정보가 다 저장되었을때
			{
				UpdateRootInfo();
			}

			bool UpdateRootInfo();

			neuro_u32 GetBlockSize() const{return m_sbInfo.nBlockSize;}

			const _NEURO_STORAGE_ALLOC_SYSTEM_ROOT& GetSystemRoot() const { return m_sbInfo.nsas_root; }

			_NEURAL_NETWORK_DEFINITION& GetNNDef() {
				return m_sbInfo.nn_def;
			}

			_NEURO_ROOT_ENTRY& GetRootEntry(){ return m_sbInfo.nn_def.root_entry; }
			void ChangeRootEntry(const _NEURO_ROOT_ENTRY& entry)
			{
				memcpy(&m_sbInfo.nn_def.root_entry, &entry, sizeof(_NEURO_ROOT_ENTRY));
			}
			
			neuro_u32 AllocBlocks(neuro_block relativeBlock, neuro_u32 nAllocBlock, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE *pAllocatedBlock);
			neuro_u32 DeallocBlocks(_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* pPNodeList, neuro_u32 nBlock);

			bool BlockIO(neuro_block block, void *pBuffer, bool bWrite);

			device::DeviceAdaptor& GetDevice(){return m_device;}

		protected:
			static neuro_u32 GetReservedBlocks(neuro_u32 group_desc_blocks_per_supergroup);

			static neuro_u32 GetGroupSize(neuro_u32 block_size);
			
			void SetNAS_SB(neuro_u32 log_block_size);

			bool ReadWriteGroupDescs(bool isWrite);

			bool AddGroup(neuro_u32 group_count = 1);

			neuro_u32 AllocBlockInGroup(neuro_u8* pBlockBitmapBuffer, neuro_u32 group_start, neuro_u32 group_last, neuro_u32 nAllocBlock, _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE *pAllocatedBlock);

			neuro_u32 GetGroupFromBlock(neuro_block block);

			neuro_block GetBlockNoFromGroupNo(neuro_u32 group_no, neuro_u32 block_index);

		private:
			device::DeviceAdaptor& m_device;

			struct _NEURO_SUPER_BLOCK_INFO
			{
				neuro_block group_start_block;
				_NEURO_STORAGE_ALLOC_SYSTEM_ROOT nsas_root;
				_NEURAL_NETWORK_DEFINITION nn_def;			// 신경망 구조 정의
				_NEURO_ALLOC_GROUP_DESCRIPTOR* group_desc_array;	// super group에 포함된 group의 desc array

				neuro_u32 nBlockSize; //block 크기(1024, 2048, 4096) 

				neuro_u32 nU32PerBlock;	// block 당 32bit 단위의 개수

				neuro_u32 group_descs_per_block;	// block size / sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR)
				neuro_u32 data_nda_per_block;		// block size / sizeof(_NEURO_DATA_ALLOC_SPEC)

				neuro_u32 groups_per_supergroup;
				neuro_u32 blocks_per_supergroup;

				neuro_u32 max_group;	// 최대 그룹 갯수
				neuro_u64 max_size;	// 최대 크기
			};
			_NEURO_SUPER_BLOCK_INFO m_sbInfo;
			void ClearSuperBlockInfo()
			{
				if (m_sbInfo.group_desc_array)
					free(m_sbInfo.group_desc_array);
				memset(&m_sbInfo, 0, sizeof(_NEURO_SUPER_BLOCK_INFO));
			}
		};
	}
}

#endif
