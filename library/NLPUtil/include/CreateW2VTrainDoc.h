#if !defined(_CREATE_W2V_TRAIN_DOC_H)
#define _CREATE_W2V_TRAIN_DOC_H

#include <string>
#include <vector>

namespace np
{
	namespace nlp
	{
		// �Է� ������ ���¼� �м���(mecab)�� ���� �Ϻ� ���(���, ����, ����� ��) ���� �ؽ�Ʈ�� �����Ͽ� ����
		struct _recv_status
		{
			size_t total_content;
//			size_t total_paragraph;
			size_t total_sentence;
			size_t total_word;

			size_t elapse;
		};
		class recv_signal
		{
		public:
			virtual void signal(const _recv_status& status) = 0;
		};

		class CreateW2VTrainDoc
		{
		public:
			CreateW2VTrainDoc();
			virtual ~CreateW2VTrainDoc();

			bool Create(const char* path, bool hasHeader, bool skip_firstline
				, bool transform_to_fastText
				, size_t split_axis, bool shuffle
				, int setup_max_words, int setup_max_sentences, int setup_max_words_per_sentence
				, std::string outfile_path, size_t flush_count = 1000, recv_signal* signal = NULL);
		};
	}
}

#endif
