#pragma once

#include "SentenceToWord.h"

#include "3rd-party/mecab/mecab-ko-v0.9.2/include/mecab.h"

namespace np
{
	namespace nlp
	{
		class MecabParser : public SentenceToWord
		{
		public:
			MecabParser(const char* rc_path, bool hasEomi = true, bool hasJosa = false);
			virtual ~MecabParser();

			bool Initialize() override;

			const char* GetRcPath() const{ return m_rc_path.c_str(); }

			bool ParseText(const char* text, const char* text_last, std::vector<_pair_string_vector>& sentence_vector) const override;

			bool ParseSentence(const std::string& sentence, _pair_string_vector& word_vector) const override;

			std::wstring GetMecabError() const;

		protected:
			bool CheckCombineSurface(const MeCab::Node* node, _word_morpheme_pair& word) const;

			const MeCab::Node* CheckNormalSurface(const MeCab::Node* node, _word_morpheme_pair& word) const;

			const MeCab::Node* CheckNormalMorphem(const MeCab::Node* node, _word_morpheme_pair& word) const;
			bool CheckSpecialMorphem(const MeCab::Node*& node, _word_morpheme_pair& pair) const;

			bool IsAllowNormalMorphem(const char* feature) const;

			static bool IsNextConn(const MeCab::Node* node);
			static bool IsNextTripleConn(const MeCab::Node* node, bool& isNumeric);
			
			bool CheckConnectSurface(const MeCab::Node*& node, _word_morpheme_pair& pair) const;

			const MeCab::Node* AppendNumericPercent(const MeCab::Node* node, _word_morpheme_pair& pair) const;

//			bool CheckCombineSurface(const MeCab::Node* node, _pair_string_vector& word_vector) const;
		private:
			static std::string GetFeature0(const char* feature);

			std::string m_rc_path;

			MeCab::Tagger *m_tagger;

			std::string m_allow_features;
		};
	}
}
