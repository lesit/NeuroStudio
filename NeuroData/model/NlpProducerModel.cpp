#include "NlpProducerModel.h"

using namespace np::dp;
using namespace np::dp::model;

NlpProducerModel::NlpProducerModel(DataProviderModel& provider, neuro_u32 uid)
	: AbstractProducerModel(provider, uid)
{
	m_use_morpheme_parser = false;
	m_use_morpheme_type_vector = false;

	m_w2v_dim = 0;
	m_is_vector_norm = false;

	m_parsing_sentence = false;
	m_max_words = 1500;
	m_max_sentences = 100;
	m_max_words_per_sentence = 500;
}

_std_u32_vector NlpProducerModel::GetUsingSourceColumnVector() const
{
	if (GetInput() == 0)
		return{};

	_std_u32_vector ret;
	
	neuro_u32 in_column_count = GetInput()->GetColumnCount();

	_u32_set::const_iterator it = m_using_colum_set.begin();
	for (; it != m_using_colum_set.end(); it++)
	{
		if (*it >= in_column_count)
			break;
		ret.push_back(*it);
	}
	return ret;
}

std::string NlpProducerModel::MakeNdfPath(const std::string& source_name) const
{
	if (GetInput() == NULL || GetInput()->GetModelType() != _model_type::reader)
		return "";

	const TextReaderModel& reader = (const TextReaderModel&)*GetInput();
	if (reader.GetReaderType() != _reader_type::text)
		return "";

	tensor::DataShape data_shape=GetDataShape();
	if (data_shape.GetDimSize() == 0)
		return "";

	std::string ndf_name = source_name;
	if (ndf_name.empty())
		return "";

	ndf_name.append(".nlp_");
	if (GetMinScale() == 0)
		ndf_name.append("0_");
	if (data_shape.GetDimSize(0, 1) > 1)
		ndf_name.append(util::StringUtil::Transform<char>(data_shape[0])).append("x");
	if (data_shape.GetDimSize(1, 2) > 1)
		ndf_name.append(util::StringUtil::Transform<char>(data_shape[1])).append("x");
	if (data_shape.GetDimSize(2, 3) > 1)
		ndf_name.append(util::StringUtil::Transform<char>(data_shape[2]));

	if (m_use_morpheme_parser)
	{
		ndf_name.append("_mp");
		if (m_use_morpheme_type_vector)
			ndf_name.append("v");
	}
	if (m_is_vector_norm)
		ndf_name.append("_vn");

	_std_u32_vector index_vector = GetUsingSourceColumnVector();
	for (int i = 0; i < index_vector.size(); i++)
	{
		ndf_name.append("_").append(util::StringUtil::Format<char>("%u", index_vector[i]));
	}
	ndf_name.append(".ndf");
	return ndf_name;
}

_ndf_dim_type NlpProducerModel::GetNdfDimType() const
{
	return _ndf_dim_type::variable_except_last;
}

void NlpProducerModel::SetNLPInfo(
	const char* mecap_rc_path
	, const char* w2v_path
	, bool use_morpheme_parser
	, bool use_morpheme_vector
	, bool is_vector_norm
	, bool parsing_sentence, neuro_u32 max_words, neuro_u32 max_sentence, neuro_u32 max_word_in_sentence)
{
	m_mecabrc_path = mecap_rc_path;
	SetWordVector(w2v_path, false);

	m_use_morpheme_parser = use_morpheme_parser;
	m_use_morpheme_type_vector = use_morpheme_vector;

	m_is_vector_norm = is_vector_norm;

	m_parsing_sentence = parsing_sentence;
	m_max_words = max_words;
	m_max_sentences = max_sentence;
	m_max_words_per_sentence = max_word_in_sentence;

	if (m_max_words==0)
		m_max_words = 1500;
	if (m_max_sentences==0)
		m_max_sentences = 100;
	if (m_max_words_per_sentence == 0 || !m_parsing_sentence && m_max_words_per_sentence>m_max_words)
		m_max_words_per_sentence = 150;

	ChangedProperty();
}

void NlpProducerModel::SetWordVector(const char* path, bool apply_changed_property)
{
	neuro_u32 words;
	if (LoadW2VHeader(path, words, m_w2v_dim))
		m_w2v_path = path;

	if(apply_changed_property)
		ChangedProperty();
}

bool NlpProducerModel::LoadW2VHeader(const char* w2v_filepath, neuro_u32& words, neuro_u32 &dim)
{
	words = dim = 0;

	np::nlp::WordToVector w2v;

	nlp::_W2V_HEADER header;
	if (!w2v.LoadHeader(w2v_filepath, header))
		return false;
	if (header.dimension == 0)
		return false;

	words = header.words;
	dim = header.dimension;
	return true;
}
