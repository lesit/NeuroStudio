#include "WordToVector.h"

#include "util/np_util.h"
#include <algorithm>

using namespace np;
using namespace np::nlp;

WordToVector::WordToVector()
:m_is_equal_fasttext_datatype(sizeof(neuron_value) == sizeof(fasttext::real))
{
	m_query_vector = NULL;
}


WordToVector::~WordToVector()
{
	delete m_query_vector;
}

_w2v_file_type WordToVector::GetFileType(const std::string& path)
{
	size_t ext = path.find_last_of('.');
	if (ext != std::string::npos && strcmp(path.c_str() + ext, ".bin") == 0)
		return _w2v_file_type::fasttext_model;
	else
		return _w2v_file_type::vector;
}

bool WordToVector::LoadHeader(const std::string& w_path, _W2V_HEADER& header)
{
	if (GetFileType(w_path) == _w2v_file_type::fasttext_model)
	{
		std::ifstream ifs(w_path, std::ifstream::binary);
		if (!ifs.is_open()) 
		{
			DEBUG_OUTPUT(L"Model file cannot be opened");
			return false;
		}

		fasttext::FastText fastText;
		if (!fastText.checkModel(ifs)) {
			DEBUG_OUTPUT(L"Model file has wrong file format!");
			return false;
		}

		fasttext::Args args;
		args.load(ifs);

		int32_t size, words;
		ifs.read((char*)&size, sizeof(int32_t));
		ifs.read((char*)&words, sizeof(int32_t));

		header.words = words;
		header.dimension = args.dim;
	}
	else
	{
		std::ifstream in(w_path);
		if (!in.is_open()) 
		{
			DEBUG_OUTPUT(L"Pretrained vectors file cannot be opened");
			return false;
		}

		in >> header.words >> header.dimension;
	}
	return true;
}

bool WordToVector::Load(const char* path)
{
	m_loaded_path.clear();

	if (m_query_vector)
		delete m_query_vector;
	m_query_vector = NULL;

	if (GetFileType(path) == _w2v_file_type::fasttext_model)
	{
		if (!m_fasttext.loadModel(path))
		{
			DEBUG_OUTPUT(L"failed loadModel[%s]", util::StringUtil::MultiByteToWide(path).c_str());
			return false;
		}
	}
	else
	{
		if (!m_fasttext.loadVectors(path, true))
		{
			DEBUG_OUTPUT(L"failed loadVectors[%s]", util::StringUtil::MultiByteToWide(path).c_str());
			return false;
		}
	}

	m_query_vector = new fasttext::Vector(m_fasttext.getDimension());

	m_loaded_path = path;
	return true;
}

int WordToVector::GetDimension() const
{
	if (!m_query_vector)
		return 0;

	return m_fasttext.getDimension();
}

bool WordToVector::HitTest(const std::string& word) const
{
	return m_fasttext.getDictionary()->getId(word) >= 0;
}

bool WordToVector::GetWordVector(bool is_norm, const std::string& word, neuro_u32 dim_count, neuron_value* vector) const
{
	if (!m_query_vector)
		return false;

	if (dim_count != m_query_vector->m_)
		return false;

	const neuro_u32 read_dim_count = std::min(m_query_vector->m_, (neuro_64)dim_count);

	m_fasttext.getVector(*m_query_vector, word);

	if (is_norm)
	{
		fasttext::real norm = m_query_vector->norm();
		if (norm > 0)
			m_query_vector->mul(1.f / norm);
	}

	if (m_is_equal_fasttext_datatype)
	{
		memcpy(vector, m_query_vector->data_, sizeof(neuron_value)*read_dim_count);
	}
	else
	{
		for (neuro_u32 i = 0; i < read_dim_count; i++)
			vector[i] = m_query_vector->data_[i];
	}

	return true;
}

bool WordToVector::GetSentenceVector(const _sentence& sentence, neuro_u32 dim_count, neuron_value* vector) const
{
	if (!m_query_vector)
		return false;

	memset(vector, 0, dim_count * sizeof(neuron_value));

	for (size_t i_word = 0, n = sentence.size(); i_word < n; i_word++)
	{
		m_fasttext.getVector(*m_query_vector, sentence[i_word]);

		fasttext::real norm = m_query_vector->norm();
		if (norm > 0)
			m_query_vector->mul(1.f / norm);

		for (size_t i = 0; i<m_query_vector->m_; i++)
			vector[i] += m_query_vector->data_[i];
	}

	if (sentence.size() > 0) 
	{
		neuro_float factor = 1.f / sentence.size();
		for (size_t i = 0; i < m_query_vector->m_; i++)
			vector[i] *= factor;
	}

	return true;
}
