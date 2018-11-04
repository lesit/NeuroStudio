#include "stdafx.h"
#include "NlpProducer.h"
#include "lib/NLPUtil/include/MecabParser.h"
#include "3rd-party/openblas-v0.2.19-64/include/cblas.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

NlpProducer::NlpProducer(const model::AbstractProducerModel& model)
	: AbstractProducer( model)
	, m_model((const model::NlpProducerModel&)model)
	, m_index_vector(m_model.GetUsingSourceColumnVector())
	, m_use_morpheme_type_vector(m_model.UseMorphemeTypeVector())
	, m_is_vector_norm(m_model.IsVectorNormalization())
	, m_parsing_sentence(m_model.ParsingSentence())
	, m_max_sentences(m_model.GetMaxSentence())
	, m_max_words_per_sentence(m_model.GetMaxWordPerSentence())
{
	// ndf mode는 문장별 channel이 있을때만 의미가 있다. 문장별 channel이 없으면 단순 2차원 배열이기 때문

	m_text_reader = NULL;
	m_w2v = NULL;
	m_s2w = NULL;

	m_w2v_dim = 0;

	m_max_content_words = 0;
#ifdef _DEBUG
	m_w2v_hit_count = m_w2v_failed_count = 0;
#endif
}

NlpProducer::~NlpProducer()
{
	DEBUG_OUTPUT(L"max word in content is %I64u", m_max_content_words);
	/*
	if (m_numeric_vector)
		free(m_numeric_vector);
		*/
}

bool NlpProducer::Create(DataReaderSet& reader_set)
{
	if (m_model.GetInput() == NULL || m_model.GetModelType() != model::_model_type::reader)
		return false;

	if (((const model::AbstractReaderModel*)m_model.GetInput())->GetReaderType() != model::_reader_type::text)
		return false;

	m_text_reader = (TextReader*)reader_set.GetReader(m_model.GetInput()->uid);
	if (m_text_reader == NULL)
		return false;

	m_w2v = reader_set.init_object.CreateW2V(m_model.GetWordToVectorPath());
	if (m_w2v == NULL)
		return false;

	m_w2v_dim = m_w2v->GetDimension();
	if (m_w2v_dim == 0)
		return false;

	m_s2w = reader_set.init_object.CreateS2W(m_model.UseMorphemeParser() ? m_model.GetMecapRcPath() : NULL);
	if (m_s2w == NULL)
		return false;

	DEBUG_OUTPUT(L"data count = %llu", m_text_reader->GetDataCount());
	return true;
}

neuro_u64 NlpProducer::GetRawDataCount() const
{
	return m_text_reader->GetDataCount();
}

neuro_u32 NlpProducer::ReadRawData(neuro_size_t pos, neuron_value* buffer, bool is_ndf_mode)
{
#ifdef _DEBUG
	if (pos == 2)
		int a = 0;
#endif

	if (!m_text_reader->Read(pos))
	{
		DEBUG_OUTPUT(L"failed readtext from test reader");
		return 0;
	}

	neuron_value* last = buffer + m_data_dim_size;
	neuron_value* ptr = buffer;

	neuro_u32* ndf_topmost_count_ptr = NULL;
	if (is_ndf_mode)	// variable_fix 이기 때문에 무조건 첫번째 dim에 대한 count를 넣어야 한다.
	{
		ndf_topmost_count_ptr = (neuro_u32*)ptr;
		*ndf_topmost_count_ptr = 0;
		((char*&)ptr) += sizeof(neuro_u32);
	}

	const neuro_u32 w2v_dim = m_w2v_dim + (m_use_morpheme_type_vector != false);

	neuro_u32 total_sentence = 0;
	neuro_u32 total_word = 0;

	for (int text_i = 0; text_i < m_index_vector.size(); text_i++)
	{
		const std::string* text = m_text_reader->GetReadText(m_index_vector[text_i]);
		if (text == NULL)
		{
			DEBUG_OUTPUT(L"no text from text reader");
			return false;

		}
		const char* paragraph = text->c_str();
		const char* text_last = text->c_str() + text->size();
		while (paragraph < text_last)
		{
			const char* paragraph_end = strchr(paragraph, '\n');
			if (paragraph == paragraph_end)
			{
				++paragraph;
				continue;
			}
			if (paragraph_end == NULL)
				paragraph_end = text_last;

			std::vector<nlp::_pair_string_vector> sentence_vector;
			if (!m_s2w->ParseText(paragraph, paragraph_end, sentence_vector))
			{
				DEBUG_OUTPUT(L"failed parse by mecab");
				return 0;
			}
			paragraph = paragraph_end + 1;

			size_t word_in_paragraph = 0;

			for (int sent_i = 0; sent_i < sentence_vector.size() && (!m_parsing_sentence || total_sentence<m_max_sentences); sent_i++)
			{
				nlp::_pair_string_vector& word_vector = sentence_vector[sent_i];

				if (word_vector.size() == 0)
					continue;

				// 문장단위로 파싱하든 아니든 문장내 최대 단어수만큼만 처리한다.
				neuro_u32 word_i = 0, word_count = min(word_vector.size(), m_max_words_per_sentence);

				// m_model.m_parsing_sentence != 0는 문장구분용 빈 단어 벡터 때문에 
				if ((ptr + (word_count + !m_parsing_sentence) * w2v_dim) > last)
				{	// 더이상 읽을게 없다.
					if (ndf_topmost_count_ptr)
						*ndf_topmost_count_ptr = m_parsing_sentence ? total_sentence : total_word;

					m_max_content_words = max(m_max_content_words, total_word);
					DEBUG_OUTPUT(L"the pointer are over. total words[%u], reading word count[%u]", total_word, word_count);
					return ptr - buffer;
				}

				if (m_parsing_sentence)
				{
					++total_sentence;
					if (is_ndf_mode)
					{
						neuro_u32* ndf_word_count_ptr = (neuro_u32*)ptr;
						*ndf_word_count_ptr = word_count;

						((char*&)ptr) += sizeof(neuro_u32);
					}
				}
				else
				{
					// 문장별로 channel을 구성하는게 아니면, 최소한 문장을 구분 할 수 있게 하자!
					// 그런데 문장의 실제 첫 단어 전에 넣은것은, rounding효과를 주기 위해서
//					memset(ptr, 0, sizeof(neuron_value)*w2v_dim);
					SetPadding(ptr, w2v_dim);

					ptr += w2v_dim;

					++total_word;
				}
				total_word += word_count;

				for (; word_i < word_count; word_i++)
				{
					if (!WordToVector(word_vector[word_i], ptr))
						return 0;
				}
#if defined(_DEBUG_NDF_TRANSFORM)
				if (m_model.m_parsing_sentence)
					DEBUG_OUTPUT(L"word count : %u", neuro_u32(word_count + !m_model.m_parsing_sentence));
#endif
				if (!is_ndf_mode && m_parsing_sentence)
				{	// 나머지 채움은 ndf 모드가 아니면서 문장별 파싱할때만 한다.
					neuro_u32 remain = (m_max_words_per_sentence - word_count) * w2v_dim;
//					memset(ptr, 0, sizeof(neuron_value)*remain);
					SetPadding(ptr, remain);

					ptr += remain;
				}
			}
		}
	}
	if (ndf_topmost_count_ptr)
		*ndf_topmost_count_ptr = m_parsing_sentence ? total_sentence : total_word;

#if defined(_DEBUG_NDF_TRANSFORM)
	DEBUG_OUTPUT(L"topmost count : %u, total sentence : %u, total word : %u\r\n"
		, ndf_topmost_count_ptr ? *ndf_topmost_count_ptr:0, total_sentence, total_word);
#endif
	m_max_content_words = max(m_max_content_words, total_word);

	return ptr - buffer;
}
#include "util/cpu_parallel_for.h"

inline bool NlpProducer::WordToVector(const np::nlp::_word_morpheme_pair& word, neuron_value*& buffer)
{
#if defined(_DEBUG)
	std::wstring str = util::StringUtil::MultiByteToWide(word.first);
	if (m_w2v->HitTest(word.first))
	{
		++m_w2v_hit_count;
	}
	else
	{
		++m_w2v_failed_count;
	}
#endif

	if (!m_w2v->GetWordVector(m_is_vector_norm, word.first, m_w2v_dim, buffer))
	{
		DEBUG_OUTPUT(L"failed GetWordVector");
		return false;
	}

	// ndf로 가져간다 하더라도 마지막은 고정이기 때문에 dim count를 굳이 적을 필요가 없다.
	// 즉, _dim_type::variable_except_last이 된다.
	buffer += m_w2v_dim;

	if (m_use_morpheme_type_vector)
	{
		static neuron_value scale = (m_scale_max - m_scale_min) / (2 * _countof(np::nlp::morphemes));	// -0.5 ~ 0.5

		neuro_u32 morpheme_id = word.getMorphemeId();
		if (morpheme_id > 0)
			*buffer = neuron_value(morpheme_id) * scale + m_scale_min / 2;
		else
			*buffer = 0;

		++buffer;
	}

	return true;
}

neuron_value* NlpProducer::FindNotSentenceToken(neuron_value* start, neuron_value* last, neuro_u32 dim, neuro_float scale_min)
{
	for (; start < last; start += dim)
	{
		int i = 0;
		for (; i < dim; i++)
		{
			if (start[i] > scale_min)
				return start;
		}
	}
	return NULL;
}

neuron_value* NlpProducer::FindSentenceToken(neuron_value* start, neuron_value* last, neuro_u32 dim, neuro_float scale_min)
{
	for (; start < last; start += dim)
	{
		int i = 0;
		for (; i < dim; i++)
		{
			if (start[i] == scale_min)
				return start;
		}
	}
	return NULL;
}
