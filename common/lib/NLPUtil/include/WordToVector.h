#pragma once

#include "3rd-party/word2vec/fastText/include/fasttext.h"

#include "SentenceToWord.h"

namespace np
{
	namespace nlp
	{
		struct _W2V_HEADER
		{
			__int64 words;
			__int64 dimension;
		};

		enum class _w2v_file_type{fasttext_model, vector};
		class WordToVector
		{
		public:
			WordToVector();
			virtual ~WordToVector();

			static _w2v_file_type GetFileType(const std::string& w_path);

			static bool LoadHeader(const std::string& w_path, _W2V_HEADER& header);

			bool Load(const char* w_path);

			const std::string& GetLoadedPath() const{ return m_loaded_path; }

			int GetDimension() const;

			bool HitTest(const std::string& word) const;

			bool GetWordVector(bool is_norm, const std::string& word, neuro_u32 dim_count, neuron_value* vector) const;

			typedef std::vector<std::string> _sentence;
			bool GetSentenceVector(const _sentence& sentence, neuro_u32 dim_count, neuron_value* vector) const;

		private:
			const bool m_is_equal_fasttext_datatype;

			std::string m_loaded_path;

			fasttext::FastText m_fasttext;
			fasttext::Vector* m_query_vector;
		};
	}
}
