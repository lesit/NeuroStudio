#if !defined(_NEURO_DATA_ALLOC_SPEC_DEFINES)
#define _NEURO_DATA_ALLOC_SPEC_DEFINES

#include "common.h"

/*	Fast Mass Storage Neural Network
*/

#pragma pack(push, 1)

namespace np
{
	namespace nsas
	{
		/*	<== 요걸로 바꾸어야 한다.
		const neuro_u8 g_alloc_btree_root_count = 7;

		struct _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE
		{
			neuro_block block_no;
		};	// 8 bytes

		static const neuro_u8 dir_blocks = 4;
		static const neuro_u8 indir_blocks = dir_blocks + 1;
		static const neuro_u8 dindir_blocks = indir_blocks + 1;
		static const neuro_u8 tindir_blocks = dindir_blocks + 1;
		union _NEURO_DATA_ALLOC_SPEC	// layer 및 layer등의 entry가 저장된 block을 정의. 54bytes
		{
			struct
			{
				neuro_u64 size;			

				// 데이터 블록(depth=0) 또는 포인터 테이블 블록
				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE block_bptree_root[g_alloc_btree_root_count];	
				//			neuro_block block_no;	// 데이터 블록(depth=0) 또는 포인터 테이블 블록
			};

			neuro_u8 reserved[64];
		};	// 64 bytes
		*/

		/*
		(block size/unit size)/2 degree B+ tree
		block size = 1 kb 일 경우
		하나의 block에 _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE를 최대 1024/8=128개 가질 수 있으므로 최소 64, 최대 128 개의 pointer를 가진다. 즉 64 degree
		*/
		const neuro_u8 g_alloc_btree_root_count = 6;

		/*	weight spec 말고는 그리 크지 않다.
		또한, weight spec은 각 입력마다 정의되기 때문에 중간 삭제/삽입을 할 경우가 없다
		왜냐하면, weight spec이 변경될 정도이면 크게 변경되기 때문에 더이상 기존 데이터가 의미 없을수도...
		차라리 처음부터 초기화하거나 추가된 개수만큼 추가하는게 나을수도.
		나머지 layer 및 layer, bias 에 대한 spec은 그리 크지 않기 때문에 더더욱 중간 삭제/삽입이 필요 없다.
		즉, 복잡하게 sub data size를 할 필요가 없다!!! 그리고, 이건 수정할때만 빠를뿐 실제 작동할땐 느리다!
		*/
		struct _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE
		{
			neuro_block block_no;
		};	// 8 bytes

		union _NEURO_DATA_ALLOC_SPEC	// layer 및 layer등의 entry가 저장된 block을 정의. 54bytes
		{
			struct
			{
				neuro_u64 size;			// 최대 4g * 4g = 16 EB

				neuro_u32 node_count;

				// 데이터 블록(depth=0) 또는 포인터 테이블 블록
				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE block_bptree_root[g_alloc_btree_root_count];
				//			neuro_block block_no;	// 데이터 블록(depth=0) 또는 포인터 테이블 블록
			};

			neuro_u8 reserved[64];
		};	// 64 bytes

			// leaf 포인터 테이블을linked list로 만들기 위해 마지막 8byte값에 next pointer table 값이 있다.
		struct _POINTER_TABLE_HEADER
		{
			neuro_u32 node_count;

			neuro_block next_pointer_table;
		};	// 12 bytes

		static neuro_u32 GetSizeInPointerTable(neuro_u32 block_size)
		{
			return block_size - sizeof(_POINTER_TABLE_HEADER);
		}

		static neuro_u32 GetPointersPerBlock(neuro_u32 block_size)
		{
			return GetSizeInPointerTable(block_size) / sizeof(neuro_block);
		}

		typedef std::vector<_NEURO_DATA_ALLOC_SPEC> _alloc_spec_vector;

		struct _POINTER_TABLE_INFO
		{
			neuro_u8* block_data;

			inline _POINTER_TABLE_HEADER* GetHeader()
			{
				if (block_data == NULL)
					return NULL;

				return (_POINTER_TABLE_HEADER*)block_data;
			}

			inline const _POINTER_TABLE_HEADER* GetHeader() const
			{
				return const_cast<_POINTER_TABLE_INFO*>(this)->GetHeader();
			}

			inline _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* GetPointerNodeList()
			{
				if (block_data == NULL)
					return NULL;

				return (_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE*)(block_data + sizeof(_POINTER_TABLE_HEADER));
			}

			inline const _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE* GetPointerNodeList() const
			{
				return const_cast<_POINTER_TABLE_INFO*>(this)->GetPointerNodeList();
			}
		};
	}
}

/*	block_no가 64bit 인경우
1 level 인 경우 : 간접 포인터
	1) 1 kb block
	128	block 표현
	즉, 128 * 1024 = 128 Kb

	2) 4 kb block 인 경우
	4kb / 8b = 512	block 표현
	즉, 512 * 4096 = 2048 kb = 2 Mb

2 level 인 경우	: 이중 간접 포인터
	1) 1 kb block
	128 * 128 = 16 k block 표현
	즉, 16 Mb

	2) 4 kb block 인 경우
	512 * 512 = 256 k block 표현
	즉, 1 Gb

3 level 인 경우 : 삼중 간접 포인터
	1) 1 kb block
	16 k * 128 = 2 m block 표현
	즉, 2 Gb

	2) 4 kb block 인 경우
	256 k * 512 = 128 m block 표현
	즉, 512 Gb
*/

/*	block_no가 64bit 이고, 마지막 leaf pointer block에 next pointer(8bytes) 가 있을 경우
	1 level 인 경우 : 간접 포인터
		1) 1 kb block
		127	block 표현
		즉, 127 * 1024 = 127 Kb

		2) 4 kb block 인 경우
		(4kb / 8b = 512) - 1 = 511 block 표현
		즉, 511 * 4096 = 2044 kb 

	2 level 인 경우	: 이중 간접 포인터
		1) 1 kb block
		128 * 127 = 15.875 block 표현
		즉, 15.875 Mb

		2) 4 kb block 인 경우
		512 * 511 = 255.5 k block 표현
		즉, 1022 Mb

	3 level 인 경우 : 삼중 간접 포인터
		1) 1 kb block
		128 * 128 * 127 = 2032 k block 표현
		즉, 2032 Mb

		2) 4 kb block 인 경우
		512 * 512 * 511 = 127.75 m block 표현
		즉, 511 Gb
*/
#pragma pack(pop)

#endif
