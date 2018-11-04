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
		/*	<== ��ɷ� �ٲپ�� �Ѵ�.
		const neuro_u8 g_alloc_btree_root_count = 7;

		struct _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE
		{
			neuro_block block_no;
		};	// 8 bytes

		static const neuro_u8 dir_blocks = 4;
		static const neuro_u8 indir_blocks = dir_blocks + 1;
		static const neuro_u8 dindir_blocks = indir_blocks + 1;
		static const neuro_u8 tindir_blocks = dindir_blocks + 1;
		union _NEURO_DATA_ALLOC_SPEC	// layer �� layer���� entry�� ����� block�� ����. 54bytes
		{
			struct
			{
				neuro_u64 size;			

				// ������ ���(depth=0) �Ǵ� ������ ���̺� ���
				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE block_bptree_root[g_alloc_btree_root_count];	
				//			neuro_block block_no;	// ������ ���(depth=0) �Ǵ� ������ ���̺� ���
			};

			neuro_u8 reserved[64];
		};	// 64 bytes
		*/

		/*
		(block size/unit size)/2 degree B+ tree
		block size = 1 kb �� ���
		�ϳ��� block�� _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE�� �ִ� 1024/8=128�� ���� �� �����Ƿ� �ּ� 64, �ִ� 128 ���� pointer�� ������. �� 64 degree
		*/
		const neuro_u8 g_alloc_btree_root_count = 6;

		/*	weight spec ����� �׸� ũ�� �ʴ�.
		����, weight spec�� �� �Է¸��� ���ǵǱ� ������ �߰� ����/������ �� ��찡 ����
		�ֳ��ϸ�, weight spec�� ����� �����̸� ũ�� ����Ǳ� ������ ���̻� ���� �����Ͱ� �ǹ� ��������...
		���� ó������ �ʱ�ȭ�ϰų� �߰��� ������ŭ �߰��ϴ°� ��������.
		������ layer �� layer, bias �� ���� spec�� �׸� ũ�� �ʱ� ������ ������ �߰� ����/������ �ʿ� ����.
		��, �����ϰ� sub data size�� �� �ʿ䰡 ����!!! �׸���, �̰� �����Ҷ��� ������ ���� �۵��Ҷ� ������!
		*/
		struct _NEURO_DATA_ALLOC_BPTREE_POINTER_NODE
		{
			neuro_block block_no;
		};	// 8 bytes

		union _NEURO_DATA_ALLOC_SPEC	// layer �� layer���� entry�� ����� block�� ����. 54bytes
		{
			struct
			{
				neuro_u64 size;			// �ִ� 4g * 4g = 16 EB

				neuro_u32 node_count;

				// ������ ���(depth=0) �Ǵ� ������ ���̺� ���
				_NEURO_DATA_ALLOC_BPTREE_POINTER_NODE block_bptree_root[g_alloc_btree_root_count];
				//			neuro_block block_no;	// ������ ���(depth=0) �Ǵ� ������ ���̺� ���
			};

			neuro_u8 reserved[64];
		};	// 64 bytes

			// leaf ������ ���̺���linked list�� ����� ���� ������ 8byte���� next pointer table ���� �ִ�.
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

/*	block_no�� 64bit �ΰ��
1 level �� ��� : ���� ������
	1) 1 kb block
	128	block ǥ��
	��, 128 * 1024 = 128 Kb

	2) 4 kb block �� ���
	4kb / 8b = 512	block ǥ��
	��, 512 * 4096 = 2048 kb = 2 Mb

2 level �� ���	: ���� ���� ������
	1) 1 kb block
	128 * 128 = 16 k block ǥ��
	��, 16 Mb

	2) 4 kb block �� ���
	512 * 512 = 256 k block ǥ��
	��, 1 Gb

3 level �� ��� : ���� ���� ������
	1) 1 kb block
	16 k * 128 = 2 m block ǥ��
	��, 2 Gb

	2) 4 kb block �� ���
	256 k * 512 = 128 m block ǥ��
	��, 512 Gb
*/

/*	block_no�� 64bit �̰�, ������ leaf pointer block�� next pointer(8bytes) �� ���� ���
	1 level �� ��� : ���� ������
		1) 1 kb block
		127	block ǥ��
		��, 127 * 1024 = 127 Kb

		2) 4 kb block �� ���
		(4kb / 8b = 512) - 1 = 511 block ǥ��
		��, 511 * 4096 = 2044 kb 

	2 level �� ���	: ���� ���� ������
		1) 1 kb block
		128 * 127 = 15.875 block ǥ��
		��, 15.875 Mb

		2) 4 kb block �� ���
		512 * 511 = 255.5 k block ǥ��
		��, 1022 Mb

	3 level �� ��� : ���� ���� ������
		1) 1 kb block
		128 * 128 * 127 = 2032 k block ǥ��
		��, 2032 Mb

		2) 4 kb block �� ���
		512 * 512 * 511 = 127.75 m block ǥ��
		��, 511 Gb
*/
#pragma pack(pop)

#endif
