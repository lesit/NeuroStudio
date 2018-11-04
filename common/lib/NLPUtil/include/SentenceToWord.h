#pragma once

#include <string>
#include <vector>

#include "util/StringUtil.h"

namespace np
{
	namespace nlp
	{
		static const char* morphemes[] = { 
			"NNG", "NNP", "NNB", "NR", "NP"
			, "VV", "VA", "VX", "VCP", "VCN"
			, "MM", "MAG", "MAJ"
			, "IC", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"
			, "EP", "EF", "EC", "ETN", "ETM"
			, "XPN", "XSN", "XSV", "XSA", "XR"
			, "SF", "SE", "SS", "SP", "SO", "SW", "SL", "SH", "SN" 
		};

		// 두번째는 tag. 없을수도 있음.
		class _word_morpheme_pair : public std::pair<std::string, std::string>
		{
		public:
			_word_morpheme_pair(){}
			_word_morpheme_pair(const std::string& first, const std::string& second)
				: pair(first, second)
			{}

			virtual ~_word_morpheme_pair(){}

			neuro_u32 getMorphemeId() const
			{
				if (second.empty())
					return 0;

				std::vector<std::string> morpheme_vector;
				if(util::StringUtil::CharsetTokenizeString(second, morpheme_vector, "+")==0)
					return 0;

				neuro_u32 max = 0;

#ifdef _DEBUG
				if (morpheme_vector.size() > 1)
					int a = 0;
#endif
				for (int i = 0, n = morpheme_vector.size(); i < n; i++)
				{
					neuro_u32 id = getMorphemeId(morpheme_vector[i].c_str());
					if (max < id)
						max = id;
				}
				return max;
			}

			static neuro_u32 getMorphemeId(const char* morphem)
			{
				if (strncmp(morphem, "NNB", 3) == 0)
					morphem = "NNB";		// mecap의 SSO, SSC -> SS
				else if (strncmp(morphem, "SS", 2) == 0)
					morphem = "SS";		// mecap의 SSO, SSC -> SS
				else if (strcmp(morphem, "SC") == 0)
					morphem = "SP";		// mecap의 SC -> SP
				else if (strcmp(morphem, "SY") == 0)
					morphem = "SW";		// mecap의 SY -> SW
				for (neuro_u32 morpheme_id = 0; morpheme_id < _countof(morphemes); morpheme_id++)
				{
					if (strcmp(morphem, morphemes[morpheme_id])==0)
						return _countof(morphemes) - morpheme_id;	// 중요한게 숫자가 높도록
				}
				return 0;
			}

			neuro_u32 morphemeScope() const
			{
				return _countof(morphemes) + 1;
			}

			std::string toString() const
			{
				std::string ret = first;
				if (!second.empty())
				{
					ret += '/';
					ret.append(second);
				}
				return ret;
			}
		};

		typedef std::vector<_word_morpheme_pair> _pair_string_vector;

		class SentenceToWord
		{
		public:
			SentenceToWord()
			{}

			virtual ~SentenceToWord(){}

			virtual bool Initialize(){
				return true;
			}

			virtual _word_morpheme_pair GetNumericMorphem() const {
				return _word_morpheme_pair("0", "SN");
			}

			virtual bool ParseText(const char* text, const char* text_last, std::vector<_pair_string_vector>& sentence_vector) const
			{
				return false;	// not ready
			}

			bool ParseSentence(const std::wstring& sentence, _pair_string_vector& word_vector) const
			{
				std::string input = util::StringUtil::WideToMultiByte(sentence);
				return ParseSentence(input, word_vector);
			}

			virtual bool ParseSentence(const std::string& sentence, _pair_string_vector& word_vector) const
			{
				const char* escape = " \r\n\t\v\f'\"";

				word_vector.clear();

				const char* str = sentence.c_str();

				size_t start = 0;
				while (start<sentence.size())
				{
					size_t end = sentence.find_first_of(escape, start);
					if (end == std::string::npos)
						end = sentence.size();

					word_vector.push_back({ std::string(str + start, str + end), "" });

					start = end + 1;
				}
				return true;
			}
		};
	}
}
