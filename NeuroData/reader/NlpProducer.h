#pragma once

#include "AbstractProducer.h"

#include "model/NlpProducerModel.h"

#include "lib/NLPUtil/include/WordToVector.h"
#include "lib/NLPUtil/include/SentenceToWord.h"

#include "TextReader.h"

namespace np
{
	namespace dp
	{
		// word embedding �ؼ� �Ǽ� ���� �о���� ���
		// �ܼ� lookup table��, word2vec/fastText ���� ����� �ܾ n���� ���� ���̺��� ����ϵ��� �Ѵ�.
		// �ϴ� �ܾ n���� ����(word-n-table �� �θ���. ��, �Ϲ� lookup table�� word-1-table�� �ȴ�.)�� �����
		// �ͺ��� �Ѵ�.!
		namespace preprocessor
		{
			class NlpProducer : public AbstractProducer
			{
			public:
				NlpProducer(const model::AbstractProducerModel& model);
				virtual ~NlpProducer();

				bool Create(DataReaderSet& reader_set) override;

				virtual const wchar_t* GetTypeString() const { return L"NlpProducer"; }

				virtual neuro_size_t GetRawDataCount() const override;
				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode = false) override;

				static neuron_value* FindNotSentenceToken(neuron_value* start, neuron_value* last, neuro_u32 dim, neuro_float scale_min);
				static neuron_value* FindSentenceToken(neuron_value* start, neuron_value* last, neuro_u32 dim, neuro_float scale_min);

			protected:
				bool WordToVector(const np::nlp::_word_morpheme_pair& word, neuron_value*& buffer);

				const model::NlpProducerModel& m_model;

				TextReader* m_text_reader;
				const np::nlp::WordToVector* m_w2v;
				const np::nlp::SentenceToWord* m_s2w;

				neuro_u32 m_w2v_dim;

				const _std_u32_vector m_index_vector;

				const bool m_use_morpheme_type_vector;
				const bool m_is_vector_norm;
				const bool m_parsing_sentence;
				const neuro_u32 m_max_sentences;
				const neuro_u32 m_max_words_per_sentence;

				neuro_size_t m_max_content_words;

#ifdef _DEBUG
				neuro_size_t m_w2v_hit_count;
				neuro_size_t m_w2v_failed_count;
#endif
			};
		}
	}
}
