#if !defined(NEURO_DATA_ACCESS_MANAGER_H)
#define NEURO_DATA_ACCESS_MANAGER_H

#include "common.h"

#include "NeuroDataAllocSpec.h"

namespace np
{
	namespace nsas
	{
		class NeuroStorageAllocationSystem;

		struct _DATA_BLOCK_INFO
		{
			bool SetDataBlockInfo(neuro_u32 nBlockSize, neuro_u64 total_size, const _POINTER_TABLE_INFO& pt_info, neuro_u32 node_index, neuro_u32 posInDataBlock);

			neuro_u32 datablock_index;
			neuro_block datablock_no;

			neuro_u32 data_size;
			neuro_u32 posInDataBlock;
		};

		struct _CACHED_DATA_BLOCK
		{
			_CACHED_DATA_BLOCK(neuro_u32 block_size)
			{
				datablock_no = 0;
				buffer = (neuro_u8*)malloc(block_size);
			}
			~_CACHED_DATA_BLOCK()
			{
				if (buffer)
					free(buffer);
			}
			neuro_block datablock_no;
			neuro_u8* buffer;
		};
		class NeuroDataAccessManager
		{
		public:
			NeuroDataAccessManager(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec);
			NeuroDataAccessManager(const NeuroStorageAllocationSystem& nsas, const _NEURO_DATA_ALLOC_SPEC& allocSpec);
			NeuroDataAccessManager(NeuroDataAccessManager& src);
			virtual ~NeuroDataAccessManager();

			NeuroStorageAllocationSystem& GetNAS(){return m_nsas;}

			bool SetDataPointer(neuro_u64 pos);
			neuro_u64 GetDataPointer() const {return m_position;}

			bool ReadData(void* buffer, neuro_u64 size) const;
			bool WriteData(const void* buffer, neuro_u64 size);

			bool SetSize(neuro_u64 size=neuro_last64);
			bool DeleteFromCurrent();

			neuro_u64 GetSize() const{return m_allocSpec.size;}

		protected:
			NeuroDataAccessManager(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec, bool read_only);

			void Initialize(NeuroStorageAllocationSystem& nsas);

			bool ReadWriteData(bool bWrite, void* buffer, neuro_u64 size);

			bool MoveNextBlock();

		private:
			NeuroStorageAllocationSystem& m_nsas;
			const bool m_bReadOnly;

			_NEURO_DATA_ALLOC_SPEC& m_allocSpec;
			neuro_u32 m_nBlockSize;

			neuro_u64 m_position;

			_POINTER_TABLE_INFO m_leaf_table;

			_DATA_BLOCK_INFO m_data_block_info;
			_CACHED_DATA_BLOCK m_cached_block;
		};
	}
}

#endif
