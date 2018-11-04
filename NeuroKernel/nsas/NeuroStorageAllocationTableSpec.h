#if !defined(_NEURO_ALLOCATION_TABLE_SPEC)
#define _NEURO_ALLOCATION_TABLE_SPEC

#include "common.h"

#include "NeuroDataAllocSpec.h"

#include "NeuralNetworkEntrySpec.h"
#include "DataProviderEntrySpec.h"

/*	Fast Mass Storage Neural Network
*/

#pragma pack(push, 1)

namespace np
{
	namespace nsas
	{
		const neuro_u32 mark_j_neuro = 0x20130517L;
		const neuro_u16 last_neuro_version[] = { 2, 0, 0, 0 };
		
		struct _NEURO_STORAGE_ALLOC_SYSTEM_ROOT
		{
			union
			{
				struct
				{
					neuro_u32 SIGNATURE;	// SIGNATURE must be mark_j_neuro
					neuro_u16 version[4];		// 2.0.0.0

					// block ũ��� group_desc_blocks_per_supergroup�� ���� �ִ�ũ�Ⱑ ��������.
					neuro_u8 log_block_size;	// entry block ũ��. kbyte���� �αװ�. default=2 : ����=BYTES_PER_KB. BYTES_PER_KB<<2 = 4 kb

					neuro_u16 blocks_per_group;	// block group�� ���Ե� block�� ����. block_size * 8

					neuro_u16 group_desc_blocks_per_supergroup;	// group desc���� ������ ����� ����. default=1
					neuro_u16 reserved_blocks_in_group;

					/*	���ǵ� super group ��.
						super group�� ��� group�� group description�� �����ϴ� ������.
						1. ���� device ���� device(hdd, ssd)
							�ϳ��� super group�� �������� �ϰ� device ũ�⿡ �´� group description ������ ���ϸ� �ȴ�.
						2. file device�� ���� ������ �þ�°� �����ؾ� �ϱ� ������ group_desc_blocks_per_supergroup�� ������ų �� ����.
							��� super_group_count �� �ø��� ������ ��� ������Ų��.
					*/
					neuro_u32 super_group_count;	
					neuro_u32 group_count;			// ���ǵ� �� ��� �׷� ��
					neuro_block blocks_count;		// ���ǵ� �� block ��
					neuro_block free_blocks_count;	// ���Ҵ�� block ��

					neuro_u32 block_group_number;	// ���� super block�� �����ϰ� �ִ� block group�� ��ȣ. ��, ����.

					neuro_block first_block_no;		// ù ��° ���, �� ��� �׷� 0�� ���۵Ǵ� ����� ����Ų��. ����, 0�̴�.
					neuro_block nn_def_block_no;	// _NEURAL_NETWORK_DEFINITION �� ����� ���. ����, 1�̴�.
					neuro_block prep_def_block_no;	// _DATA_PREPROCESS �� ����� ���. ����, 1�̴�.
				};

				neuro_u8 reserved[128];
			};
		};	// �� ũ��� ���� 1024 bytes �� ���� �ʾƾ� �Ѵ�. �ּ� block ũ�Ⱑ 1kb �̹Ƿ�

		struct _NEURO_DATA_ALLOC_TABLE
		{
			neuro_u32 free_nda_count;
			_NEURO_DATA_ALLOC_SPEC bitmap_spec;	// nid�� ���� bitmap

			/*	nid�� _NEURO_DATA_ALLOC_SPEC���� �����ϴ� spec.
				��, n ��° nda�� n * sizeof(_NEURO_DATA_ALLOC_SPEC) ��ġ�� �ִ�.
			*/
			_NEURO_DATA_ALLOC_SPEC nda_spec;
		};

		// GoolLeNet �� �� 95���� layer�� ������ ������, 63 ���� weight�� ������ layer(conv, fc) �� ������ �ִ�.
		// �̷��� ���� �ƴϸ�, HiddenLayerEntry�� ���θ鼭 ���� ������(multi input, weight ��)�� ���� �ִ� 4���� _NEURO_DATA_ALLOC_SPEC ���� ������ �ִ� ����ü�� ��������.
		// �׷��� �ϸ�, �ϳ��� HiddenLayerEntry �� 256byte�� ����. 100,000 ���� layer�� ��� 24mb ����. �� ���� �� ũ��??
		struct _NEURAL_NETWORK_DEFINITION
		{
			_NEURO_ROOT_ENTRY root_entry;	// 512

			union
			{
				struct
				{
					// layer nda spec�� ��� layer�� �ϰ������� �����ϴ� ������� �Ѵ�.
					_NEURO_DATA_ALLOC_SPEC input_layer_nda;	// input layer entry�� �Ҵ��ϱ� ���� spec. �����͸� �Է¹޴� ���̾��̱� ������ ���� �и���
					_NEURO_DATA_ALLOC_SPEC hidden_layer_nda;	// hidden layer entry�� �Ҵ��ϱ� ���� spec

					// nda table�� �ʿ��Ҷ� �Ҵ��ϰ� �ʿ��������(��, layer���� �����Ǹ�) �Ҵ� �����ϴ� ������� �Ѵ�.
					// writing�Ҷ� �ϴ� nda_table�� �о� ���� �Ŀ� �����Ұ� �����ϰ�, �߰��Ҷ� free �� �ͺ��� ä�������� id�� ��� �����ϰ�, ���̻� ������ table�� ������Ű�� �ȴ�.
					// �׸���, ���� ������ nid�� table�� ���������� �ؼ� ũ�� ���� �ϸ� �ȴ�.

					// multi input, weight, bias, �Ϲ� ������ ���� ���� _NEURO_DATA_ALLOC_SPEC �Ҵ��� ���� table
					// �ϳ��� layer���� weight�� �����ϴ� �����. weight�� ���� nid�� nda_table_spec �κ��� ���� _NEURO_DATA_ALLOC_SPEC �� ������ �ͼ� ����.
					// ��, ������ layer(layer) entry �� �ִ� _NEURO_DATA_ALLOC_SPEC ���ٴ� �Ѵܰ� �� ����. ������, entry�� ũ��� �پ���.
					_NEURO_DATA_ALLOC_TABLE nda_table;
				};

				neuro_u8 reserved[512];
			};
		};	// 1024 bytes

		// ���� �̰� ���⿡�� �� �ʿ䰡 ������... ��εȴ�... �ϴ� ���߿� �����غ���
		struct _DATA_PROVIDER_DEF
		{
			union _DATA_PREPROCESS_DEF
			{
				_NEURO_DATA_ALLOC_SPEC reader_nda;
				_NEURO_DATA_ALLOC_SPEC producer_nda;

				_NEURO_DATA_ALLOC_TABLE nda_table;
			};
		};

		/*	16 byte�̱� ������ 1block���� 1 block size / 16 ���� group�� �����Ƿ�,
			block size�� 1kb �� ��� 1024/16 = 64���� group �� ���� �� �ִ�.
			*/
		union _NEURO_ALLOC_GROUP_DESCRIPTOR
		{
			struct
			{
				neuro_u32 bitmap_block_index;	// bitmap block index in group
				neuro_u32 free_blocks_count;
			};

			neuro_u8 reserved[16];
		};	// 16bytes
	}
}

#pragma pack(pop)

/*
	�ΰ��� ���� 1000�ﰳ�� ������ ����. �ϳ��� ������ �ּ� 100������ �ִ� 10000���� ������ ������ ����

	1000���� layer�� �ִٰ� �����Ѵٸ� �� layer �� 100,000,000(1��) ���� neuron�� �ִ� ���̴�.
	1,000���� ������ �ϳ��� �׷����� ���´ٸ� �ϳ��� layer�� 100,000 ���� �׷��� �ִٰ� �����Ҽ� �ִ�.(�׳��� �־��� �ó�����)

	1. 1000���� layer�� ���� �Ҵ� ũ��� 128b * 1000 = �� 128 kb
	2. �ϳ��� layer�� ǥ���ϱ� ���� �Ҵ� ũ��� 256bytes �̹Ƿ� �ϳ��� layer�� ���Ե� layer���� �Ҵ� ũ��� 256 * 100,000 = 25,600,000 = �� 24.4 MB 
	���� ��� layer�� �Ҵ� ũ��� 24.4mb * 1000 = �� 23.8 GB
	3. 1000�ﰳ�� ������ �Ҵ� ũ��� 128 * 100,000,000,000 = 11.6TB
	4. �� ������ 10,000���� ������ �����ٰ� �����ϸ� 
	�ϳ��� ���� �� weight�� ���� �Ҵ� ũ�� = sizeof(neuro_float) * 10000 = 8 * 10000 = �� 80 KB
	1000�ﰳ�� ������ weight�� ���� �Ҵ� ũ�� = 80kb * 100,000,000,000 = 7.275 EB

*/

/*	block group �̾߱�
	�ϳ��� block group�� �ϳ��� bitmap table block�� ������.
	�ϳ��� bitmap table block�� ǥ�������� ���� ���� 1024*2^log_block_size*8 ���̴�. ��, block ũ�� * 8 �̴�.
	�̰��� �ϳ��� byte�� 8bit�� �����Ǿ� �־ block ũ�� * 8bit ��ŭ�� ���� ǥ���� �� �ֱ� �����̴�.
	��, �ϳ��� block group�� block size * 8 ���� block�� ������ �ִ�.
	
	����, �ϳ��� block group�� ũ��� block size * 8 * block size = 8 * block_size^2

	1kb¥�� bitmap table. �� 1024 * 8 = 8192 block ǥ��
	block�� ũ�Ⱑ 1kb�� ��� 8192 block�� ǥ���ϹǷ�, �ϳ��� block group�� ũ��� 8192 kb
	block�� ũ�Ⱑ 4kb�� ��� 32,768 block ǥ��. ����, �ϳ��� block group�� ũ��� 32,768 * 4 kb = 128 mb 
	block�� ũ�Ⱑ 64kb�� ��� 524,288 block ǥ��. ����, �ϳ��� block group�� ũ��� 524,288 block * 64 kb = 32 gb 

	group descriptor table�� ũ�Ⱑ 1 block �� ��� : group�� ���� = block size / group descriptor size(16bytes)
	����, block_group_count = group_desc_blocks_per_supergroup(block count of group descriptor table) * block_size / 16
*/
/*	��ġ ũ�⺰ block group ����. 
	1TB�� Neuro Memory�� ǥ���ϱ� ���ؼ�
		- block�� ũ�Ⱑ 1kb�� ��� 1 giga���� block ���ǰ� �ʿ�. �ϳ��� block bitmap�� 1024*8[1byte�� 8bit�̹Ƿ�]���� ��� ����. ��, �ϳ��� group�� 1024*8���� block�� ������.
			1g/8k = 128 k ���� block group �ʿ�.
			����, �� 16bytes[sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR)] * 128k = 2 mb ũ���� group descryptor �ʿ��ϸ� �ʿ��� block���� 2mb / 1k = 2 k ��
		- block�� ũ�Ⱑ 4kb�� ��� 256 mega���� block ���ǰ� �ʿ�. �ϳ��� group�� 4*1024*8���� block�� ������.
			256m/32k = 8 k���� block group �ʿ�.
			����, �� 16bytes * 8k = 128 kb ũ���� group descryptor �ʿ��ϸ� �ʿ��� block���� 128kb / 4kb = 32 ��
		- block�� ũ�Ⱑ 64kb�� ��� 16 mega���� block ���ǰ� �ʿ�. �ϳ��� group�� 64*1024*8���� block�� ������.
			16m/512k =  32���� block group �ʿ�.
			����, �� 16bytes * 32 = 516 b ũ���� group descryptor �ʿ��ϸ� �ʿ��� block���� 516b / 64kb = 1 ��
*/

/*	super group �̾߱�

	���� ��ġ ũ�⺰ block group count ����� �����ϸ� ������ ����.
	block_group_count = (total_size / block_size) / block_size*8 = total_size / (block_size^2*8)

	������ block group ������ ���� description�� �����ϴ� ũ��� description ũ�Ⱑ 16 byte�̹Ƿ�
	total_group_desc_size = 16 * block_group_count

	����, �� description���� �����ϱ� ���� block �� ������
	group_desc_blocks_per_supergroup=total_group_desc_size/block_size
	��, 	total_group_desc_size = group_desc_blocks_per_supergroup * block_size

	total_group_desc_size�� ġȯ�ϸ�
	16 * block_group_count = group_desc_blocks_per_supergroup * block_size
	block_size = 16 * block_group_count / group_desc_blocks_per_supergroup
	block_size = 16 * total_size / (block_size^2*8) / group_desc_blocks_per_supergroup
	block_size^3 = 2 * total_size / group_desc_blocks_per_supergroup

	����, �� �޸� ũ���
	total_size = group_desc_blocks_per_supergroup * block_size^3 / 2

	���屸���� �� block group�� ������
	block_group_count	= total_size / (block_size^2*8)
						= (group_desc_blocks_per_supergroup * block_size^3 / 2) / (block_size^2 * 8)
						= group_desc_blocks_per_supergroup * block_size^3 / (block_size^2 * 16)
						= group_desc_blocks_per_supergroup * block_size / 16

	1. ����, group_descriptor�� 1���� block�� �����Ѵٸ�
		1) block ũ�Ⱑ 1Kb �� ���
			total_size = 1g / 2 = 512m �� �ȴ�.
			512mb�� 512mb/1k = 512k ���� block�� ���ǵǰ�, �ϳ��� block bitmap�� 8k ���� ����� �����ϹǷ�
			512k/8k=64 ���� block group�� ���ǵȴ�. �� ���� block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 1 * 1024/16 ���� ��ġ�Ѵ�.
			��, 16b*64=1kb ũ���� group descriptor�� �ʿ��ϸ� �� ũ��� ��Ȯ�� �ϳ��� block ũ�Ⱑ �ȴ�.

		2) block ũ�Ⱑ 4Kb �ΰ��
			total_size = 64g / 2 = 32g �� �ȴ�.
			32Gb�� 32Gb/4k = 8m ���� block�� ���ǵǰ�, �ϳ��� block bitmap�� 32k ���� ����� �����ϹǷ�
			8m/32k=256 ���� block group�� ���ǵȴ�. �� ���� block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 1 * 4k/16 ���� ��ġ�Ѵ�.
			��, 16b*256=4kb ũ���� group descriptor�� �ʿ��ϸ� �� ũ��� ��Ȯ�� �ϳ��� block ũ�Ⱑ �ȴ�.

	2. ����, group_descriptor�� 8���� block�� �����Ѵٸ�
		1) block ũ�Ⱑ 1Kb �� ���
			total_size = 8 * 1G / 2 = 4Gb �� �ȴ�.
			4Gb�� 4Gb/1k = 4m ���� block�� ���ǵǰ�, �ϳ��� block bitmap�� 8k ���� ����� �����ϹǷ�
			4m/8k=512 ���� block group�� ���ǵȴ�. �� ���� block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 8 * 1k/16 ���� ��ġ�Ѵ�.

		2) block ũ�Ⱑ 4Kb �ΰ��
			total_size = 8 * 64g / 2 = 256g �� �ȴ�.
			256g�� 256g/4k = 64m ���� block�� ���ǵǰ�, �ϳ��� block bitmap�� 32k ���� ����� �����ϹǷ�
			64m/32k=2k ���� block group�� ���ǵȴ�. �� ���� block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 8 * 4k/16 ���� ��ġ�Ѵ�.
*/

#endif
