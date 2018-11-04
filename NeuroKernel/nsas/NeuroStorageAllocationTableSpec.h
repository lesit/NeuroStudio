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

					// block 크기와 group_desc_blocks_per_supergroup에 따라 최대크기가 정해진다.
					neuro_u8 log_block_size;	// entry block 크기. kbyte단위 로그값. default=2 : 단위=BYTES_PER_KB. BYTES_PER_KB<<2 = 4 kb

					neuro_u16 blocks_per_group;	// block group에 포함된 block의 개수. block_size * 8

					neuro_u16 group_desc_blocks_per_supergroup;	// group desc들을 저장할 블록의 개수. default=1
					neuro_u16 reserved_blocks_in_group;

					/*	정의된 super group 수.
						super group는 모든 group은 group description을 공유하는 단위다.
						1. 고정 device 고정 device(hdd, ssd)
							하나의 super group만 가지도록 하고 device 크기에 맞는 group description 개수를 정하면 된다.
						2. file device는 거의 무한정 늘어나는걸 지원해야 하기 때문에 group_desc_blocks_per_supergroup을 고정시킬 수 없다.
							대신 super_group_count 를 늘리는 식으로 계속 증가시킨다.
					*/
					neuro_u32 super_group_count;	
					neuro_u32 group_count;			// 정의된 총 블록 그룹 수
					neuro_block blocks_count;		// 정의된 총 block 수
					neuro_block free_blocks_count;	// 비할당된 block 수

					neuro_u32 block_group_number;	// 현재 super block을 포함하고 있는 block group의 번호. 즉, 원본.

					neuro_block first_block_no;		// 첫 번째 블록, 즉 블록 그룹 0이 시작되는 블록을 가리킨다. 물론, 0이다.
					neuro_block nn_def_block_no;	// _NEURAL_NETWORK_DEFINITION 가 저장된 블록. 물론, 1이다.
					neuro_block prep_def_block_no;	// _DATA_PREPROCESS 가 저장된 블록. 물론, 1이다.
				};

				neuro_u8 reserved[128];
			};
		};	// 이 크기는 절대 1024 bytes 가 넘지 않아야 한다. 최소 block 크기가 1kb 이므로

		struct _NEURO_DATA_ALLOC_TABLE
		{
			neuro_u32 free_nda_count;
			_NEURO_DATA_ALLOC_SPEC bitmap_spec;	// nid에 대한 bitmap

			/*	nid의 _NEURO_DATA_ALLOC_SPEC들을 저장하는 spec.
				즉, n 번째 nda는 n * sizeof(_NEURO_DATA_ALLOC_SPEC) 위치에 있다.
			*/
			_NEURO_DATA_ALLOC_SPEC nda_spec;
		};

		// GoolLeNet 은 총 95개의 layer를 가지고 있으며, 63 개의 weight을 가지는 layer(conv, fc) 를 가지고 있다.
		// 이렇게 할지 아니면, HiddenLayerEntry를 감싸면서 각종 데이터(multi input, weight 등)을 위한 최대 4개의 _NEURO_DATA_ALLOC_SPEC 들을 가지고 있는 구조체로 저장할지.
		// 그렇게 하면, 하나의 HiddenLayerEntry 당 256byte가 증가. 100,000 개의 layer의 경우 24mb 증가. 뭐 별로 안 크네??
		struct _NEURAL_NETWORK_DEFINITION
		{
			_NEURO_ROOT_ENTRY root_entry;	// 512

			union
			{
				struct
				{
					// layer nda spec은 모든 layer를 일괄적으로 저장하는 방식으로 한다.
					_NEURO_DATA_ALLOC_SPEC input_layer_nda;	// input layer entry를 할당하기 위한 spec. 데이터를 입력받는 레이어이기 때문에 따로 분리함
					_NEURO_DATA_ALLOC_SPEC hidden_layer_nda;	// hidden layer entry를 할당하기 위한 spec

					// nda table은 필요할때 할당하고 필요없어지면(즉, layer등이 삭제되면) 할당 해제하는 방식으로 한다.
					// writing할때 일단 nda_table을 읽어 들인 후에 삭제할건 삭제하고, 추가할때 free 인 것부터 채워나가며 id를 얻어 재사용하고, 더이상 없으면 table을 증가시키면 된다.
					// 그리고, 최종 마지막 nid을 table의 마지막으로 해서 크기 조정 하면 된다.

					// multi input, weight, bias, 일반 데이터 등의 하위 _NEURO_DATA_ALLOC_SPEC 할당을 위한 table
					// 하나의 layer에서 weight에 접근하는 방법은. weight에 대한 nid로 nda_table_spec 로부터 실제 _NEURO_DATA_ALLOC_SPEC 을 가지고 와서 접근.
					// 즉, 기존에 layer(layer) entry 에 있는 _NEURO_DATA_ALLOC_SPEC 보다는 한단계 더 접근. 하지만, entry의 크기는 줄어든다.
					_NEURO_DATA_ALLOC_TABLE nda_table;
				};

				neuro_u8 reserved[512];
			};
		};	// 1024 bytes

		// 굳이 이걸 여기에서 할 필요가 있을까... 고민된다... 일단 나중에 생각해보자
		struct _DATA_PROVIDER_DEF
		{
			union _DATA_PREPROCESS_DEF
			{
				_NEURO_DATA_ALLOC_SPEC reader_nda;
				_NEURO_DATA_ALLOC_SPEC producer_nda;

				_NEURO_DATA_ALLOC_TABLE nda_table;
			};
		};

		/*	16 byte이기 때문에 1block에는 1 block size / 16 개의 group이 있으므로,
			block size가 1kb 인 경우 1024/16 = 64개의 group 을 가질 수 있다.
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
	인간의 뇌엔 1000억개의 뉴런이 있음. 하나의 뉴런은 최소 100개에서 최대 10000개의 연결을 가지고 있음

	1000개의 layer가 있다고 가정한다면 각 layer 당 100,000,000(1억) 개의 neuron이 있는 것이다.
	1,000개의 뉴런을 하나의 그룹으로 묶는다면 하나의 layer에 100,000 개의 그룹이 있다고 가정할수 있다.(그나마 최악의 시나리오)

	1. 1000개의 layer에 대한 할당 크기는 128b * 1000 = 약 128 kb
	2. 하나의 layer을 표현하기 위한 할당 크기는 256bytes 이므로 하나의 layer에 포함된 layer들의 할당 크기는 256 * 100,000 = 25,600,000 = 약 24.4 MB 
	따라서 모든 layer의 할당 크기는 24.4mb * 1000 = 약 23.8 GB
	3. 1000억개의 뉴런의 할당 크기는 128 * 100,000,000,000 = 11.6TB
	4. 각 뉴런당 10,000개의 연결을 가진다고 가정하면 
	하나의 뉴런 당 weight에 대한 할당 크기 = sizeof(neuro_float) * 10000 = 8 * 10000 = 약 80 KB
	1000억개의 뉴런의 weight에 대한 할당 크기 = 80kb * 100,000,000,000 = 7.275 EB

*/

/*	block group 이야기
	하나의 block group은 하나의 bitmap table block을 가진다.
	하나의 bitmap table block이 표현가능한 블럭의 수는 1024*2^log_block_size*8 개이다. 즉, block 크기 * 8 이다.
	이것은 하나의 byte가 8bit로 구성되어 있어서 block 크기 * 8bit 만큼의 블럭을 표현할 수 있기 때문이다.
	즉, 하나의 block group은 block size * 8 개의 block을 가지고 있다.
	
	따라서, 하나의 block group의 크기는 block size * 8 * block size = 8 * block_size^2

	1kb짜리 bitmap table. 총 1024 * 8 = 8192 block 표현
	block의 크기가 1kb인 경우 8192 block을 표현하므로, 하나의 block group의 크기는 8192 kb
	block의 크기가 4kb인 경우 32,768 block 표현. 따라서, 하나의 block group의 크기는 32,768 * 4 kb = 128 mb 
	block의 크기가 64kb인 경우 524,288 block 표현. 따라서, 하나의 block group의 크기는 524,288 block * 64 kb = 32 gb 

	group descriptor table의 크기가 1 block 일 경우 : group의 개수 = block size / group descriptor size(16bytes)
	따라서, block_group_count = group_desc_blocks_per_supergroup(block count of group descriptor table) * block_size / 16
*/
/*	장치 크기별 block group 예제. 
	1TB의 Neuro Memory를 표현하기 위해선
		- block의 크기가 1kb인 경우 1 giga개의 block 정의가 필요. 하나의 block bitmap은 1024*8[1byte가 8bit이므로]개의 블록 정의. 즉, 하나의 group은 1024*8개의 block을 가진다.
			1g/8k = 128 k 개의 block group 필요.
			따라서, 총 16bytes[sizeof(_NEURO_ALLOC_GROUP_DESCRIPTOR)] * 128k = 2 mb 크기의 group descryptor 필요하며 필요한 block수는 2mb / 1k = 2 k 개
		- block의 크기가 4kb인 경우 256 mega개의 block 정의가 필요. 하나의 group은 4*1024*8개의 block을 가진다.
			256m/32k = 8 k개의 block group 필요.
			따라서, 총 16bytes * 8k = 128 kb 크기의 group descryptor 필요하며 필요한 block수는 128kb / 4kb = 32 개
		- block의 크기가 64kb인 경우 16 mega개의 block 정의가 필요. 하나의 group은 64*1024*8개의 block을 가진다.
			16m/512k =  32개의 block group 필요.
			따라서, 총 16bytes * 32 = 516 b 크기의 group descryptor 필요하며 필요한 block수는 516b / 64kb = 1 개
*/

/*	super group 이야기

	위의 장치 크기별 block group count 계산을 정리하면 다음과 같다.
	block_group_count = (total_size / block_size) / block_size*8 = total_size / (block_size^2*8)

	임의의 block group 개수에 대한 description을 저장하는 크기는 description 크기가 16 byte이므로
	total_group_desc_size = 16 * block_group_count

	따라서, 이 description들을 저장하기 위한 block 의 개수는
	group_desc_blocks_per_supergroup=total_group_desc_size/block_size
	즉, 	total_group_desc_size = group_desc_blocks_per_supergroup * block_size

	total_group_desc_size를 치환하면
	16 * block_group_count = group_desc_blocks_per_supergroup * block_size
	block_size = 16 * block_group_count / group_desc_blocks_per_supergroup
	block_size = 16 * total_size / (block_size^2*8) / group_desc_blocks_per_supergroup
	block_size^3 = 2 * total_size / group_desc_blocks_per_supergroup

	따라서, 총 메모리 크기는
	total_size = group_desc_blocks_per_supergroup * block_size^3 / 2

	저장구조의 총 block group의 개수는
	block_group_count	= total_size / (block_size^2*8)
						= (group_desc_blocks_per_supergroup * block_size^3 / 2) / (block_size^2 * 8)
						= group_desc_blocks_per_supergroup * block_size^3 / (block_size^2 * 16)
						= group_desc_blocks_per_supergroup * block_size / 16

	1. 만약, group_descriptor를 1개의 block에 저장한다면
		1) block 크기가 1Kb 인 경우
			total_size = 1g / 2 = 512m 가 된다.
			512mb는 512mb/1k = 512k 개의 block이 정의되고, 하나의 block bitmap은 8k 개의 블록을 정의하므로
			512k/8k=64 개의 block group이 정의된다. 이 값은 block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 1 * 1024/16 값과 일치한다.
			즉, 16b*64=1kb 크기의 group descriptor가 필요하며 이 크기는 정확히 하나의 block 크기가 된다.

		2) block 크기가 4Kb 인경우
			total_size = 64g / 2 = 32g 가 된다.
			32Gb는 32Gb/4k = 8m 개의 block이 정의되고, 하나의 block bitmap은 32k 개의 블록을 정의하므로
			8m/32k=256 개의 block group이 정의된다. 이 값은 block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 1 * 4k/16 값과 일치한다.
			즉, 16b*256=4kb 크기의 group descriptor가 필요하며 이 크기는 정확히 하나의 block 크기가 된다.

	2. 만약, group_descriptor를 8개의 block에 저장한다면
		1) block 크기가 1Kb 인 경우
			total_size = 8 * 1G / 2 = 4Gb 가 된다.
			4Gb는 4Gb/1k = 4m 개의 block이 정의되고, 하나의 block bitmap은 8k 개의 블록을 정의하므로
			4m/8k=512 개의 block group이 정의된다. 이 값은 block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 8 * 1k/16 값과 일치한다.

		2) block 크기가 4Kb 인경우
			total_size = 8 * 64g / 2 = 256g 가 된다.
			256g는 256g/4k = 64m 개의 block이 정의되고, 하나의 block bitmap은 32k 개의 블록을 정의하므로
			64m/32k=2k 개의 block group이 정의된다. 이 값은 block_group_count = group_desc_blocks_per_supergroup * block_size / 16 = 8 * 4k/16 값과 일치한다.
*/

#endif
