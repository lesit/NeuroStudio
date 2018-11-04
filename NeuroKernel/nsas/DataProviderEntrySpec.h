#if !defined(_DATA_PROVIDER_ENTRY_SPEC_H)
#define _DATA_PROVIDER_ENTRY_SPEC_H

#include "common.h"

#pragma pack(push, 1)

namespace np
{
	namespace nsas
	{
		struct _BIN_CONTENT_COL_INFO
		{
			char name[16];
			neuro_u32 bytes;
		};	// 20 bytes

		struct _BINARY_READER
		{
			_BIN_CONTENT_COL_INFO bin_col_vector[20];
		};	// 400 bytes

		struct _TEXT_CONTENT_COL_DELIMITER
		{
			char token_vector[8][8];// csv���� "," "\t"
			neuro_u32 fixed_len;
		};	// 68 bytes

		struct _TEXT_CONTENT_COL_INFO
		{
			char name[16];
		};	// 16 bytes

		struct _TEXT_READER
		{
			char content_delimiter[8];					// content�� token csv���� \n

			_TEXT_CONTENT_COL_DELIMITER column_delimiter;	// content�������� column �м��� ���� ������.		68 bytes
			bool has_double_qoute;						// "    " ���� �ν�. ������ csv ����� , �� column�� ���������� "   " �ε� �����Ѵ�.

			_TEXT_CONTENT_COL_INFO col_vector[20];	// �ִ� 20���� column reading ����.	320 bytes
		};

		union _DATA_READER_ENTRY
		{
			struct
			{
				neuro_u32 uid;
				neuro_u16 type;

				neuro_u32 input_uid;

				neuro_u32 column_nid;
				union
				{
					_BINARY_READER bin;
					_TEXT_READER text;
				};
			};
			char reserved[512];
		};

		struct _NUMERIC_PRODUCER_ENTRY
		{
			struct _COL_INDEX_DEF
			{
				_COL_INDEX_DEF()
				{
					src_column = 0;
					ma = 1;
				}
				_COL_INDEX_DEF(neuro_u32 src_column, neuro_u32 ma = 1)
				{
					this->src_column = src_column;
					this->ma = ma;
				}

				neuro_u32 src_column;	// data filter������ index. �ּ� 0�̻�
				neuro_u32 ma;		// moving average. �ּ� 1�̻�
			};

			_COL_INDEX_DEF column_vector[16];
		};

		struct _NLP_PRODUCER_ENTRY
		{
			char morphem_parser_rc_filename[64];
			bool use_morpheme_vector;

			char w2v_filename[64];

			neuro_u32 source_index_vector[16];

			bool is_vector_norm;

			neuro_u32 w2v_dim;			// �ܾ n���� ���� ���̺��� ���� ����. loading �Ҷ��� �����ϴ�.

			bool parsing_sentence;
			neuro_u32 max_words;
			neuro_u32 max_sentences;		// time�� �� ���̴�. ��, time seriese�� ó���Ͽ� conv net �ϳ��ε� �����ϵ���
			neuro_u32 max_words_per_sentence;
		};

		struct _LIB_PRODUCER_ENTRY
		{
			neuro_u16 type;	// _library_type
			char filename[255];// path root�� system�� ���� ����
		};

		union _DATA_PRODUCER_ENTRY
		{
			struct
			{
				neuro_u32 uid;
				neuro_u16 type;	// _producer_type

				neuro_u32 input_uid;
				union
				{
					_NUMERIC_PRODUCER_ENTRY numeric;
					_NLP_PRODUCER_ENTRY nlp;
					_LIB_PRODUCER_ENTRY lib;
					neuro_u8 entry_size[256];
				};
				bool use_ndf_on_train;
			};
			neuro_u8 reserved[64];
		};
	}
}
#pragma pack(pop)

#endif
