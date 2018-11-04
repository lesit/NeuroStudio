#include "MecabParser.h"

#include "util/np_util.h"
#include "util/StringUtil.h"

using namespace np;
using namespace np::nlp;
/*
	2017.11.23 ���� 4��
	�� �ȵǴٰ� �ƴ�!
	���¼� �м��� ���� �ε�. �ʹ� �и��Ǿ� �ִ°� �׳� ����ȭ�ߴ�. �׷����� fastText���� �ν��ϴ� �ܾ���� �� �þ���
	���ڿ� , �� �ִ°� ������ �ذ���
	�׸��� �Ʒÿ� ������ ���鶧 ���� �״�� �������. ���� ���� ���ڰ� �����°� �־�����.
	�׸��� trim �Լ��� �߸� �ƾ��� �Ѥ�
	����, ç�������� ���� ���Ϸ� �м� ��� �������� ����� �ܾ���� �پ���.
	�׷���, �� �ܾ���� 1500���� ���̰�, �ִ� ����� 100�� ����� �ִ� �ܾ���� 150���� ���ϼ� �־���
*/
MecabParser::MecabParser(const char* rc_path, bool hasEomi, bool hasJosa)
{
	m_rc_path = rc_path;

	m_tagger = NULL;

	m_allow_features = { 'N', 'V', 'M', 'I', 'X' };
	if (hasJosa)
		m_allow_features.append("J");
	if (hasEomi)
		m_allow_features.append("E");
}


MecabParser::~MecabParser()
{
	if (m_tagger)
		MeCab::deleteTagger(m_tagger);
}

bool MecabParser::Initialize()
{
	if (m_rc_path.empty())
		return false;

	if (m_tagger)
		MeCab::deleteTagger(m_tagger);

	std::string arg = "-r ";
	arg.append(m_rc_path);
	m_tagger = MeCab::createTagger(arg.c_str());
	if (!m_tagger)
	{// ������ ��� mecabrc ������ ã�� ���ϴ� �����μ�, debugging�� ��� �۾� ���丮�� '$(ProjectDir)\bin\$(OutDir)' �̷��� �ؾ��Ѵ�!
		DEBUG_OUTPUT(L"failed create mecab tagger. %s", GetMecabError().c_str());
		MeCab::deleteTagger(m_tagger);

		return false;
	}
	return true;
}

bool MecabParser::ParseText(const char* text, const char* text_last, std::vector<_pair_string_vector>& sentence_vector) const
{
	sentence_vector.clear();
	if (text >= text_last)
		return true;

	if (!m_tagger)
	{
		DEBUG_OUTPUT(L"no tagger");
		return false;
	}

	//tagger.parse(input.c_str());
	const MeCab::Node* node = m_tagger->parseToNode(text, text_last-text);
	if (!node)
	{
		DEBUG_OUTPUT(L"failed parseToNode. %s", GetMecabError().c_str());
		return false;
	}

	sentence_vector.resize(1);
	_pair_string_vector* word_vector = &sentence_vector[sentence_vector.size() - 1];

//#define _WSTRING_ORG_DEBUG
#if defined(_DEBUG) & defined(_WSTRING_ORG_DEBUG)
	std::wstring wsentence;
	util::StringUtil::MultiByteToWide(text, text_last-text, wsentence);
	DEBUG_OUTPUT(L"origi : %s", wsentence.c_str());

	if (wsentence.find(L"�������� 36.4%") != std::wstring::npos)
		int a = 0;
#endif

	for (; node; node = node->next)
	{
		const wchar_t* node_format = L"";
		switch (node->stat)
		{
		case MECAB_BOS_NODE:		// Virtual node representing a beginning of the sentence.
			node_format = L"bos";
			break;
		case MECAB_EOS_NODE:		// Virtual node representing a end of the sentence.
			node_format = L"eos";
			break;
		case MECAB_UNK_NODE:
			node_format = L"unk";	// Unknown node not defined in the dictionary.
			break;
		case MECAB_NOR_NODE:
			node_format = L"nor";	// Normal node defined in the dictionary.
			break;
		case MECAB_EON_NODE:		// Virtual node representing a end of the N-best enumeration.
			node_format = L"eon";
			break;
		}
		if (node->stat == MECAB_BOS_NODE || node->stat == MECAB_EOS_NODE)
			continue;

//		#define _WSTRING_CUR_DEBUG
#if defined(_DEBUG) & defined(_WSTRING_CUR_DEBUG)
		std::wstring wfeature = util::StringUtil::MultiByteToWide(GetFeature0(node->feature));
		std::wstring wsurface = util::StringUtil::MultiByteToWide(std::string(node->surface, node->length));
#endif
		if (strncmp(node->feature, "SF", 2) == 0)
		{
			// �׳� ������ ù��°�� ���ڷ� ���еǴ� �Ŷ��.. 1. 2. 3. �� ����
			// �Ǵ� �߸� �Ľ̵Ǿ� �������� 1. 2. 3. �� ���� ���. ���� �ϳ��� ����̹Ƿ�.
			if (word_vector->size() > 0)
			{
				if (word_vector->back().second == "SN")
					word_vector->erase(word_vector->end() - 1);
			}

			if (word_vector->size() == 0)	// ��������� �� ������ ���� �ʿ�� �����Ƿ�
				continue;

			sentence_vector.resize(sentence_vector.size() + 1);
			word_vector = &sentence_vector.back();
			continue;
		}

		_word_morpheme_pair word;
		if (!CheckCombineSurface(node, word))	// �������� ���
		{
			node = CheckNormalSurface(node, word);	// �������¼�(ü��, ���, ���ľ�, ������ ��)
		}

		if (!word.first.empty())
		{
			if (util::StringUtil::CheckUtf8((unsigned char*)word.first.c_str(), word.first.size()) == util::_str_utf8_type::failure)
				DEBUG_OUTPUT(L"failed check utf8. %s", util::StringUtil::MultiByteToWide(word.first).c_str());

			word_vector->push_back(word);
			/*
			// ������ EC�� ����. ���� '��' '���ϴ�' �̷��͵� �̴�!
			if (word.second != "EC")
				*/
		}
#if defined(_DEBUG) & defined(_WSTRING_CUR_DEBUG)
		else
			DEBUG_OUTPUT(L"skiped feature[%s], surface[%s]", wfeature.c_str(), wsurface.c_str());
#endif
	}

	if (sentence_vector.size() > 0)
	{
		const _pair_string_vector& word_vector = sentence_vector[sentence_vector.size() - 1];
		if (word_vector.size() == 0 || word_vector.size() == 1 && word_vector[0].second == "SN")
			sentence_vector.erase(sentence_vector.begin() + sentence_vector.size() - 1);

	}
//	#define _WSTRING_PARSED_DEBUG
#if defined(_DEBUG) & defined(_WSTRING_PARSED_DEBUG)
	for(int i=0;i<sentence_vector.size();i++)
	{
		_pair_string_vector& word_vector=sentence_vector[i];
		std::wstring str;
		str = util::StringUtil::MultiByteToWide(word_vector[0].first);
		for (int i = 1; i < word_vector.size(); i++)
		{
			str.append(L" ");
			str.append(util::StringUtil::MultiByteToWide(word_vector[i].first));
		}
		DEBUG_OUTPUT(L"words : %s", str.c_str());
	}
#if defined(_DEBUG) & defined(_WSTRING_ORG_DEBUG)
	DEBUG_OUTPUT(L"");
#endif
#endif
	return true;
}

bool MecabParser::ParseSentence(const std::string& sentence, _pair_string_vector& word_pair_vector) const
{
	word_pair_vector.clear();
	if (sentence.size() == 0)
		return true;

	if (!m_tagger)
	{
		DEBUG_OUTPUT(L"no tagger");
		return false;
	}

	//tagger.parse(input.c_str());
	const MeCab::Node* node = m_tagger->parseToNode(sentence.c_str(), sentence.size());
	if (!node)
	{
		DEBUG_OUTPUT(L"failed parseToNode. %s", GetMecabError().c_str());
		return false;
	}

#if defined(_DEBUG) & defined(_WSTRING_ORG_DEBUG)
	std::wstring wsentence = util::StringUtil::MultiByteToWide(sentence);
	DEBUG_OUTPUT(L"origi : %s", wsentence.c_str());
#endif

	for (; node; node = node->next)
	{
		const wchar_t* node_format = L"";
		switch (node->stat)
		{
		case MECAB_BOS_NODE:		// Virtual node representing a beginning of the sentence.
			node_format = L"bos";
			break;
		case MECAB_EOS_NODE:		// Virtual node representing a end of the sentence.
			node_format = L"eos";
			break;
		case MECAB_UNK_NODE:
			node_format = L"unk";	// Unknown node not defined in the dictionary.
			break;
		case MECAB_NOR_NODE:
			node_format = L"nor";	// Normal node defined in the dictionary.
			break;
		case MECAB_EON_NODE:		// Virtual node representing a end of the N-best enumeration.
			node_format = L"eon";
			break;
		}
		if (node->stat == MECAB_BOS_NODE || node->stat == MECAB_EOS_NODE)
			continue;

//#define _WSTRING_CUR_DEBUG
#if defined(_DEBUG) & defined(_WSTRING_CUR_DEBUG)
		std::wstring wfeature = util::StringUtil::MultiByteToWide(GetFeature0(node->feature));
		std::wstring wsurface = util::StringUtil::MultiByteToWide(std::string(node->surface, node->length));
#endif
		/*
		if (CheckCombineSurface(node, word_pair_vector))	// �������� ���
			continue;
			*/
		_word_morpheme_pair word;
		if (!CheckCombineSurface(node, word))	// �������� ���
		{
			node = CheckNormalSurface(node, word);	// �������¼�(ü��, ���, ���ľ�, ������ ��)
		}

		if (!word.first.empty())
		{
			if (util::StringUtil::CheckUtf8((unsigned char*)word.first.c_str(), word.first.size()) == util::_str_utf8_type::failure)
				DEBUG_OUTPUT(L"failed check utf8. %s", util::StringUtil::MultiByteToWide(word.first).c_str());

			word_pair_vector.push_back(word);
		}
#if defined(_DEBUG) & defined(_WSTRING_CUR_DEBUG)
		else
			DEBUG_OUTPUT(L"feature[%s], surface[%s]", wfeature.c_str(), wsurface.c_str());
#endif

//#define _SOME_FIND
#ifdef _SOME_FIND
		std::wstring temp = util::StringUtil::MultiByteToWide(word.first);
		if (temp == L"��Ű")
			int a = 0;
		if (word.second == "XSN" || word.second == "XSV" || word.second == "XSN" || word.second == "ETN" || word.second == "ETM")
		{
			DEBUG_OUTPUT(std::wstring(L"test : ").append(util::StringUtil::MultiByteToWide(word.toString())).c_str());
		}
#endif

	}

	if (word_pair_vector.size() > 0)
	{
		// ������ EC�� ����. ���� '��' '���ϴ�' �̷��͵� �̴�!
		if (word_pair_vector[word_pair_vector.size() - 1].second == "EC")
			word_pair_vector.erase(word_pair_vector.begin() + word_pair_vector.size() - 1);
	}

#if defined(_DEBUG) & defined(_WSTRING_PARSED_DEBUG)
	if (word_pair_vector.size() > 0)
	{
		std::wstring str;
		str = util::StringUtil::MultiByteToWide(word_pair_vector[0].first);
		for (int i = 1; i < word_pair_vector.size(); i++)
		{
			str.append(L" ");
			str.append(util::StringUtil::MultiByteToWide(word_pair_vector[i].first));
		}
		DEBUG_OUTPUT(L"words : %s", str.c_str());
	}
#if defined(_WSTRING_ORG_DEBUG)
	DEBUG_OUTPUT(L"");
#endif
#endif
	return true;
}

bool MecabParser::IsAllowNormalMorphem(const char* feature) const
{
	if (*(feature + 1) == '\0')
		return false;

	char cur_ch = feature[0];
	for (size_t i = 0, n = m_allow_features.size(); i < n; i++)
	{
		if (cur_ch != m_allow_features[i])
			continue;

		/*	2017.11.22 �ʹ� ���ص� �ؼ� �ٽ� �־���.
		if (feature[0] == 'V')
		{
			if (feature[1] == 'C' && feature[2] == 'P')	// ������������(VCP)�� �����ϰ� ������(����������� VCN����)�� ���
				continue;
		}
		else if (feature[0] == 'E')	// ����϶�
		{
			// EP(���), EF(����), EC(����)�� CheckCombineSurface ���� ó�� �ϵ��� �Ѵ�.
			if (feature[1] == 'P' || feature[1] == 'F')// || feature[1] == 'C')	������ ���ϱ� �ʹ� �̻�
				continue;
		}
		*/
		return true;
	}
	return false;
}

bool MecabParser::CheckCombineSurface(const MeCab::Node* node, _word_morpheme_pair& word) const
{
	const char* p = node->feature;
	for (; *p != '\0' && *p != ',' && *p != '+'; p++);

	if (*p != '+')
		return false;
	++p;
	if (*p == '\0')
		return false;

	// ü���� �ƴϸ鼭 combine �� ���¼� ������ ã�� ����.
	// �� ������ , ������ ã�´�.
	const char* small_surface = p + strlen(p) - 1;
	for (; small_surface > p && *small_surface != ','; small_surface--);

	if (small_surface <= p)
		return false;

	++small_surface;

	while (*small_surface != '\0')
	{
		const char* small_surface_end = strchr(small_surface, '/');
		if (small_surface_end == NULL)
		{
			DEBUG_OUTPUT(L"strange!. small surface is zero");
			break;
		}

		const char* small_feature = small_surface_end + 1;
		const char* small_feature_end = small_feature;
		for (; *small_feature_end != '\0' && *small_feature_end != '/'; small_feature_end++);
		if (small_feature_end == small_feature)
		{
			DEBUG_OUTPUT(L"strange!. small feature is zero");
			break;
		}

		if (IsAllowNormalMorphem(small_feature))
		{
			//	2017.11.22 �ʹ� �������ؼ� �׳� ���Ǵ°� �ϳ��� ������ �и����� �ʰ� �ֵ��� �Ѵ�.
			word.first.assign(node->surface, node->surface + node->length);
			word.second = GetFeature0(node->feature);
			return true;
		}

		small_surface = small_feature_end + 1;
		for (; *small_surface != '\0' && *small_surface != '+'; small_surface++);

		if (*small_surface == '\0')
			break;

		++small_surface;
	}
	return true;
}

const MeCab::Node* MecabParser::CheckNormalSurface(const MeCab::Node* node, _word_morpheme_pair& word) const
{
	if (CheckConnectSurface(node, word))
	{
		if (word.second == "SN")
			node = AppendNumericPercent(node, word);

		return node;
	}

	if (CheckSpecialMorphem(node, word))
		return node;

	if (IsAllowNormalMorphem(node->feature))
		return CheckNormalMorphem(node, word);

	return node;
}

/*	���ӵǴ� surface ó��
N N
NNG/NNP/SL/SH/SN_NNG/NNP/SL/SH/SN
NNG/NNP/SL/SH/SN.NNG/NNP/SL/SH/SN
SN,SN
SN%
�� ÷��
XPN N
+SN
-SN
*/
bool MecabParser::CheckConnectSurface(const MeCab::Node*& node, _word_morpheme_pair& word) const
{
	bool isNumeric;
	if (!IsNextTripleConn(node, isNumeric))
		return false;

	if (isNumeric)
	{
		if (word.second.empty())
			word.second = "SN";
	}
	else
		word.second = "NNP";

	word.first.append(node->surface, node->surface + node->length);
	word.first.append(node->next->surface, node->next->surface + node->next->length);

	node = node->next->next;
	if (!CheckConnectSurface(node, word))
	{
		word.first.append(node->surface, node->surface + node->length);
	}
	return true;
}

const MeCab::Node* MecabParser::CheckNormalMorphem(const MeCab::Node* node, _word_morpheme_pair& word) const
{
	word.first.append(node->surface, node->surface + node->length);
	word.second = GetFeature0(node->feature);

	if (IsNextConn(node))
	{
		if (node->feature[0] == 'N')
		{
			if (node->next->feature[0] == 'N')
				return CheckNormalMorphem(node->next, word);

			// XPN�� ���λ��µ� �и� ���� surface�ٷ� �ڿ� �پ��µ� XPN�̶�� �ϴ� ��찡 �ִ� �Ѥ�
			// �ܿ��� -> �ܿ�/NN ��/XPN �̶�� ���� �Ѥ�
			// �̰� ó�� ���ߴ��� ������ �ƴ�! ���߿� ���� ���캸��!
			if (strncmp(node->next->feature, "XPN", 3) == 0 || strcmp(GetFeature0(node->next->feature).c_str(), "UNKNOWN")==0)
			{
				//#define _WSTRING_PARSED_BUG_DEBUG
#ifdef _WSTRING_PARSED_BUG_DEBUG
				std::wstring str;
				util::StringUtil::MultiByteToWide(node->surface, node->length, str);

				std::wstring next_str;
				util::StringUtil::MultiByteToWide(node->next->surface, node->next->length, next_str);

				DEBUG_OUTPUT(L"stupid mecab parsed. nn[%s], xpn[%s]", str.c_str(), next_str.c_str());
#endif

				_word_morpheme_pair next_pair;
				node = CheckNormalMorphem(node->next, next_pair);

				word.first.append(next_pair.first);
				return node;
			}
		}
		else if (strncmp(node->feature, "XPN", 3) == 0 && IsAllowNormalMorphem(node->next->feature))
		{
			return CheckNormalMorphem(node->next, word);
		}
	}
	return node;
}

bool MecabParser::CheckSpecialMorphem(const MeCab::Node*& node, _word_morpheme_pair& word) const
{
	if (node->feature[0] != 'S' || *(node->feature + 1) == '\0')
		return false;

	// SL(�ܱ���), SH(����), SN(����) �� ����Ѵ�.
	const char f_char1 = node->feature[1];
	if (f_char1 == 'N')	// ������ ���
	{
		word.second = GetFeature0(node->feature);
		word.first.append(node->surface, node->surface + node->length);
		node = AppendNumericPercent(node, word);

		return true;
	}

	if (f_char1 == 'L' || f_char1 == 'H')	// �ܱ���, ���� �ΰ��
	{
		word.first.assign(node->surface, node->surface + node->length);
		word.second = GetFeature0(node->feature);
		return true;
	}

	// ���� ��ȣ�� ���� �ǹ̰� ������ ������ ���ش�.
	const bool has_sign = true;
	if (f_char1 == 'Y' && node->length == 1)
	{
		const char sy_char = node->surface[0];

		if (has_sign && IsNextConn(node) && strchr("+-", sy_char) != NULL && strncmp(node->next->feature, "SN", 2) == 0)
		{
			_word_morpheme_pair next_word;
			node = CheckNormalSurface(node->next, next_word);
			if (next_word.second == "SN")
			{
				word.first = sy_char;
				word.first.append(next_word.first);
				word.second = next_word.second;
			}
			return true;
		}

		// ���ڿ� �������ִ� ���ڱ�ȣ�� ���� �ҿ� ������ ����..
		const char* allow_special_chars = "+-*/<>&|%"; // ���б�ȣ�� ���
		if (node->prev && GetFeature0(node->prev->feature) == "SN" && strchr(allow_special_chars, sy_char) != NULL)
		{
			word.first.append(node->surface, node->surface + 1);
			word.second = GetFeature0(node->feature);
			return true;
		}
	}
	return true;
}

inline std::string MecabParser::GetFeature0(const char* feature)
{
	const char* feature_end = feature;
	for (; *feature_end != '\0' && *feature_end != ','; feature_end++);

	return std::string(feature, feature_end);
}

inline bool MecabParser::IsNextConn(const MeCab::Node* node)
{
	if (!node->next || node->next->length == 0)
		return false;

	if (node->surface + node->length != node->next->surface)
		return false;

	return true;
}

inline bool MecabParser::IsNextTripleConn(const MeCab::Node* node, bool& isNumeric)
{
	if (!IsNextConn(node) || !IsNextConn(node->next))
		return false;

	const MeCab::Node* mid_node = node->next;
	if (mid_node->length != 1)
		return false;
	
	if (strncmp(node->feature, "SN", 2) == 0 && strncmp(mid_node->next->feature, "SN", 2) == 0)
	{
		if (mid_node->surface[0] == '.' && strncmp(mid_node->feature, "SY", 2) == 0
			|| mid_node->surface[0] == ',' && strncmp(mid_node->feature, "SC", 2) == 0)
		{
			isNumeric = true;
			return true;
		}
	}

	if (strncmp(mid_node->feature, "SY", 2) != 0)
		return false;

	if (strchr("_.@", mid_node->surface[0]) == NULL)	// .�� @ �� url
		return false;

	isNumeric = false;

	const char* allow_side_feature[] = { "NN", "SL", "SH", "SN" };
	int i = 0, n = _countof(allow_side_feature);
	for (; i < n; i++)
	{
		if (strncmp(node->feature, allow_side_feature[i], 2) == 0)
			break;
	}
	if (i == n)
		return false;

	for (i = 0; i < n; i++)
	{
		if (strncmp(mid_node->next->feature, allow_side_feature[i], 2) == 0)
			break;
	}
	if (i == n)
		return false;

	return true;
}

const MeCab::Node* MecabParser::AppendNumericPercent(const MeCab::Node* node, _word_morpheme_pair& pair) const
{
	if (node->next && strncmp(node->next->feature, "SY", 2) == 0 && node->next->surface[0] == '%')
	{
		pair.first.append("%");
		return node->next;
	}
	return node;
}

std::wstring MecabParser::GetMecabError() const
{
	const char *e = m_tagger ? m_tagger->what() : MeCab::getTaggerError();
	if (!e)
		return L"";
	return std::wstring(e, e + strlen(e));
}

/*
//	�̰� 2017.1121 ������ �ҽ�
bool MecabParser::CheckCombineSurface(const MeCab::Node* node, _pair_string_vector& word_vector) const
{
#ifdef _DEBUG
	if (strncmp(node->feature, "VV+EC", 5) == 0)
		int a = 0;
#endif

	const char* p = node->feature;
	for (; *p != '\0' && *p != ',' && *p != '+'; p++);

	if (*p != '+')
		return false;
	++p;
	if (*p == '\0')
		return false;

	// ü���� �ƴϸ鼭 combine �� ���¼� ������ ã�� ����.
	// �� ������ , ������ ã�´�.
	const char* small_surface = p + strlen(p) - 1;
	for (; small_surface > p && *small_surface != ','; small_surface--);

	if (small_surface <= p)
		return false;

	++small_surface;

	while (*small_surface != '\0')
	{
		const char* small_surface_end = strchr(small_surface, '/');
		if (small_surface_end == NULL)
		{
			DEBUG_OUTPUT(L"strange!. small surface is zero");
			break;
		}

		const char* small_feature = small_surface_end + 1;
		const char* small_feature_end = small_feature;
		for (; *small_feature_end != '\0' && *small_feature_end != '/'; small_feature_end++);
		if (small_feature_end == small_feature)
		{
			DEBUG_OUTPUT(L"strange!. small feature is zero");
			break;
		}

		if (IsAllowNormalMorphem(small_feature))
		{
			word_vector.resize(word_vector.size() + 1);
			_word_morpheme_pair& word = word_vector[word_vector.size() - 1];

			word.first.assign(small_surface, small_surface_end);
			word.second.assign(small_feature, small_feature_end);
		}

		small_surface = small_feature_end + 1;
		for (; *small_surface != '\0' && *small_surface != '+'; small_surface++);

		if (*small_surface == '\0')
			break;

		++small_surface;
	}
	return true;
}
*/