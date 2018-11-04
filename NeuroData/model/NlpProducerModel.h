#pragma once

#include "AbstractProducerModel.h"
#include "TextReaderModel.h"

#include "lib/NLPUtil/include/WordToVector.h"
#include "lib/NLPUtil/include/SentenceToWord.h"

namespace np
{
	namespace dp
	{
		// word embedding 해서 실수 값을 읽어오는 기능
		// 단순 lookup table과, word2vec/fastText 등을 사용한 단어별 n차원 벡터 테이블을 사용하도록 한다.
		// 일단 단어별 n차원 벡터(word-n-table 로 부르자. 즉, 일반 lookup table은 word-1-table이 된다.)를 사용한
		// 것부터 한다.!

		namespace model
		{
			class NlpProducerModel : public AbstractProducerModel
			{
			public:
				NlpProducerModel(DataProviderModel& provider, neuro_u32 uid);
				virtual ~NlpProducerModel() {}

				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::must; }

				_producer_type GetProducerType() const override {
					return _producer_type::nlp;
				}
				bool AvailableToOutputLayer() const override { return false; }

				std::string MakeNdfPath(const std::string& origin_file_path) const override;

				bool SupportNdfClone() const override { return true; }
				_ndf_dim_type GetNdfDimType() const override;

				virtual tensor::DataShape GetDataShape() const override {
					if (m_using_colum_set.empty())
						return tensor::DataShape();

					const bool use_morpheme_vector = m_use_morpheme_parser ? m_use_morpheme_type_vector : false;
					return MakeDataShape(m_w2v_dim, use_morpheme_vector, m_parsing_sentence, m_max_words, m_max_sentences, m_max_words_per_sentence);
				}

				static inline tensor::DataShape MakeDataShape(neuro_u32 word_dim, bool use_morpheme_vector
					, bool parsing_sentence, neuro_u32 max_words, neuro_u32 max_sentences, neuro_u32 max_words_in_sentence)
				{
					if (parsing_sentence)
						return tensor::DataShape({ max_sentences, max_words_in_sentence, word_dim + use_morpheme_vector });
					else// max_sentences 를 더하는 이유는 문장 구분때문
						return tensor::DataShape({ 1, max_words + max_sentences, word_dim + use_morpheme_vector });
				}

				static bool LoadW2VHeader(const char* w2v_filepath, neuro_u32& words, neuro_u32 &dim);

				void SetNLPInfo(
					const char* mecap_rc_path
					, const char* w2v_path
					, bool use_morpheme_parser
					, bool use_morpheme_vector
					, bool is_vector_norm
					, bool parsing_sentence, neuro_u32 max_words, neuro_u32 max_sentence, neuro_u32 max_word_in_sentence);

				void SetMecapRcPath(const char* path) { m_mecabrc_path = path; }

				void SetWordVector(const char* path, bool apply_changed_property=true);

				const char* GetMecapRcPath() const { return m_mecabrc_path.c_str(); }
				const char* GetWordToVectorPath() const { return m_w2v_path.c_str(); }

				void SetUseMorphemeParser(bool use) 
				{
					m_use_morpheme_parser=use; 
					ChangedProperty();
				}
				bool UseMorphemeParser() const { return m_use_morpheme_parser; }

				void SetUseMorphemeTypeVector(bool use) { 
					m_use_morpheme_type_vector = use; 
					ChangedProperty();
				}
				bool UseMorphemeTypeVector() const { return m_use_morpheme_type_vector; }

				neuro_u32 GetWordDimension() const { return m_w2v_dim; }

				void SetVectorNormalization(bool is_norm) { m_is_vector_norm = is_norm; }
				bool IsVectorNormalization() const { return m_is_vector_norm; }

				void SetParsingSentence(bool is) {
					m_parsing_sentence = is; 
					ChangedProperty();
				}
				bool ParsingSentence() const { return m_parsing_sentence; }

				void SetMaxWord(neuro_u32 max) {
					m_max_words = max; 
					ChangedProperty();
				}
				neuro_u32 GetMaxWord() const { return m_max_words; }

				void SetMaxSentence(neuro_u32 max) {
					m_max_sentences = max; 
					ChangedProperty();
				}
				neuro_u32 GetMaxSentence() const { return m_max_sentences; }

				void SetMaxWordPerSentence(neuro_u32 max) { 
					m_max_words_per_sentence = max; 
					ChangedProperty();
				}
				neuro_u32 GetMaxWordPerSentence() const { return m_max_words_per_sentence; }

				void ChangedProperty() override
				{
					if (GetInput())
					{
						neuro_u32 max_column_count = GetInput()->GetColumnCount();
						_u32_set::const_reverse_iterator it = m_using_colum_set.rbegin();
						for (; it != m_using_colum_set.rend(); it++)
						{
							if (*it >= max_column_count)
								m_using_colum_set.erase(*it);
							else
								break;
						}
					}
					else
					{
						m_using_colum_set.clear();
					}
					__super::ChangedProperty();
				}

				const _u32_set& GetUsingSourceColumnSet() const { return m_using_colum_set; }
				void InsertSourceColumn(neuro_u32 column) 
				{
					bool old_empty = m_using_colum_set.empty();

					m_using_colum_set.insert(column); 
					if (old_empty)
						ChangedProperty();
				}
				void EraseSourceColumn(neuro_u32 column) {
					bool old_has = !m_using_colum_set.empty();

					m_using_colum_set.erase(column);
					if (old_has && m_using_colum_set.empty())
						ChangedProperty();
				}

				_std_u32_vector GetUsingSourceColumnVector() const;
				void GetUsingSourceIndexSet(_u32_set& index_set) const override
				{
					index_set.insert(m_using_colum_set.begin(), m_using_colum_set.end());
				}

			private:
				_u32_set m_using_colum_set;

				std::string m_mecabrc_path;
				std::string m_w2v_path;

				bool m_use_morpheme_parser;
				bool m_use_morpheme_type_vector;
				bool m_is_vector_norm;

				neuro_u32 m_w2v_dim;			// 단어별 n차원 벡터 테이블의 차원 개수. loading 할때만 가능하다.

				bool m_parsing_sentence;
				neuro_u32 m_max_words;
				neuro_u32 m_max_sentences;		// time이 될 놈이다. 즉, time seriese로 처리하여 conv net 하나로도 가능하도록
				neuro_u32 m_max_words_per_sentence;
			};
		}
	}
}
