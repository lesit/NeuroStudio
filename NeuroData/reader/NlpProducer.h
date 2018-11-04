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
		// word embedding 해서 실수 값을 읽어오는 기능
		// 단순 lookup table과, word2vec/fastText 등을 사용한 단어별 n차원 벡터 테이블을 사용하도록 한다.
		// 일단 단어별 n차원 벡터(word-n-table 로 부르자. 즉, 일반 lookup table은 word-1-table이 된다.)를 사용한
		// 것부터 한다.!
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
