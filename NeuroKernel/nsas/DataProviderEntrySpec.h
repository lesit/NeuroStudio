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
			char token_vector[8][8];// csv에선 "," "\t"
			neuro_u32 fixed_len;
		};	// 68 bytes

		struct _TEXT_CONTENT_COL_INFO
		{
			char name[16];
		};	// 16 bytes

		struct _TEXT_READER
		{
			char content_delimiter[8];					// content의 token csv에선 \n

			_TEXT_CONTENT_COL_DELIMITER column_delimiter;	// content내에서의 column 분석을 위한 구분자.		68 bytes
			bool has_double_qoute;						// "    " 구분 인식. 보통의 csv 방식은 , 로 column을 구분하지만 "   " 로도 구분한다.

			_TEXT_CONTENT_COL_INFO col_vector[20];	// 최대 20개의 column reading 지원.	320 bytes
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

				neuro_u32 src_column;	// data filter에서의 index. 최소 0이상
				neuro_u32 ma;		// moving average. 최소 1이상
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

			neuro_u32 w2v_dim;			// 단어별 n차원 벡터 테이블의 차원 개수. loading 할때만 가능하다.

			bool parsing_sentence;
			neuro_u32 max_words;
			neuro_u32 max_sentences;		// time이 될 놈이다. 즉, time seriese로 처리하여 conv net 하나로도 가능하도록
			neuro_u32 max_words_per_sentence;
		};

		struct _LIB_PRODUCER_ENTRY
		{
			neuro_u16 type;	// _library_type
			char filename[255];// path root는 system에 직접 전달
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
