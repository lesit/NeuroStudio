#include "stdafx.h"

#include "TextContentParser.h"

using namespace np;
using namespace np::dnp;

void TextContentParser::ParseContent(const std::string& paragraph_token, const std::vector<std::string>& text_vector, _content& content)
{
	content.resize(text_vector.size());
	for (size_t i = 0, n = text_vector.size(); i < n; i++)
	{
		ParseText(paragraph_token, text_vector[i], content[i]);
	}
}

inline void TextContentParser::ParseText(const std::string& paragraph_token, const std::string& str, _content_col& text)
{
	size_t i = 0;
	while (i < str.size())
	{
		size_t found = str.find(paragraph_token, i);

		size_t end = found == std::string::npos ? str.size() : found;

		if (i < end)
		{
			text.resize(text.size() + 1);
			_paragraph& paragraph = text[text.size() - 1];
			ParseParagraph(std::string(str.c_str() + i, str.c_str() + end), paragraph);
		}

		if (found == std::string::npos)
			break;

		i = found + paragraph_token.size();
	}
}

inline void TextContentParser::ParseParagraph(const std::string& str, _paragraph& paragraph)
{
	paragraph.clear();

#ifdef _DEBUG
	std::wstring wstr = util::StringUtil::MultiByteToWide(str);
	if (wstr.find(L"ü��") != std::wstring::npos)
		int a = 0;
#endif
	const char* start = str.c_str();
	const char* end = start + str.size();
	while (start < end)
	{
		const char* found = start;
		for (; found < end; found++)
		{
			if (*found == '?' || *found == '!')
			{
				for (; found < end; found++)	// ?? ?! !! ������ �� ã�� ���ؼ�
				{
					if (*found != '.' && *found != '?' && *found != '!')
						break;
				}
				--found;
				break;
			}
			if (*found == '.')
			{
				const char* end_sentenc_token = found + 1;
				for (; end_sentenc_token < end && *end_sentenc_token == '.'; end_sentenc_token++);
				if (end_sentenc_token>found + 1)	// .. ... ������ �ɷ����� ���ؼ�
				{
					found = end_sentenc_token + 1;	// �̰� ��ħǥ�� �ƴѰ� Ȯ���ϴϱ� �������� �̵�
					continue;
				}
				break;
			}
		}

		if (found == end)
			break;

		if (found == start)
		{
			++start;
			continue;
		}

		AddSentence(paragraph, start, found + 1);	// ��ħǥ�� ? �� ! �� ���Խ�Ű��!
		start = found + 1;
	}

	if (start < end)
		AddSentence(paragraph, start, end);

	ArrangeSentenceVector(paragraph);
}

// sentence�� . �� ���еǴ°ſ� ��ȿ
void TextContentParser::AddSentence(_paragraph& paragraph, const char* start, const char* last)
{
	if (last == start)
		return;

	if (paragraph.size() > 0)
	{
#ifdef _DEBUG
		std::wstring wstr = util::StringUtil::MultiByteToWide(paragraph[paragraph.size() - 1]);

#endif
		const char* p = start;

		for (; p < last; p++)
		{
			const char* white = util::default_word_escapes;
			for (; *white != '\0' && *p != *white; white++);

			if (*white == '\0')
				break;
		}

		for (; p < last; p++)
		{
			if (!isdigit(*p))
				break;
		}

		// �� ���� �߰�! ���� ���ڹٷ� �ڿ� ���ڰ� ������ ��ħǥ��� ���������� �ƴҼ���.. �׷��� ��
		if (p>start)// && !util::StringUtil::IsLangChar(p, p+1))
		{
			_sentence& last_sentence = paragraph[paragraph.size() - 1];

			const char* prev_start = last_sentence.c_str();
			const char* prev_last = prev_start + last_sentence.size();
			const char* p = prev_last - 1;
			if (p >= prev_start && *p == '.')	// �Ҽ����ϼ��� ���� ���
			{
				--p;
				for (; p >= prev_start; p--)
				{
					if (!isdigit(*p))
						break;
				}

				// ���� ���ڹٷ� �ڿ� ���ڰ� ������ ��ħǥ��� ���������� �ƴҼ���.. �׷��� ��
				//			if (p < prev_start || (p < prev_last - 1 && !util::StringUtil::IsLangChar(p, p + 1)))
				if (p < prev_start || p < prev_last - 1)
				{
					// ����� ���ڴ�!
					last_sentence.append(".").append(start, last);
					return;
				}
			}
		}
	}
	if (start < last)
	{
		std::string sentence = std::string(start, last);
		if (sentence.size() == 1)
			int a = 0;
		util::StringUtil::Trim(sentence);

		if (!sentence.empty())
			paragraph.push_back(sentence);
		else
			int a = 0;
	}
}

void TextContentParser::ArrangeSentenceVector(_paragraph& paragraph)
{
	_paragraph::iterator it = paragraph.begin();
	for (; it != paragraph.end(); it++)
	{
		it = ArrangeSentenceVectorForEmail(paragraph, it);
		if (it == paragraph.end())
			break;
		it = ArrangeSentenceVectorForWeb(paragraph, it);
		if (it == paragraph.end())
			break;

		// source package url�� ó�� ���Ѵ�. ���� ���, oracle.util. ���
		// ���߿� �����غ���!
	}
}

inline _paragraph::iterator TextContentParser::ArrangeSentenceVectorForEmail(_paragraph& paragraph, _paragraph::iterator it)
{
	_sentence sent = *it;	// ������ �ϸ� �Ʒ� erase�� ���� �����ǰ� �ȴ�.

	const char found_string[] = "@";

	size_t found = sent.find(found_string);
	if (found == std::string::npos)
		return it;

	it = ArrangeSentencesStartURL(paragraph, it, sent.c_str(), sent.c_str() + found);
	it = ArrangeSentencesEndURL(paragraph, it, sent.c_str() + found + _countof(found_string));
	return it;
}

inline _paragraph::iterator TextContentParser::ArrangeSentenceVectorForWeb(_paragraph& paragraph, _paragraph::iterator it)
{
	_sentence sent = *it;	// ������ �ϸ� �Ʒ� erase�� ���� �����ǰ� �ȴ�.

	const char found_string[] = "://";

	size_t found = sent.find(found_string);
	if (found == std::string::npos)
		return it;

	return ArrangeSentencesEndURL(paragraph, it, sent.c_str() + found + _countof(found_string));
}

inline _paragraph::iterator TextContentParser::ArrangeSentencesStartURL(_paragraph& paragraph, _paragraph::iterator it, const char* start, const char* last)
{
	// �տ� ���ڳ� ���ڰ� �ƴѰ� �ϳ��� �����ٸ�, �� ���忡�� ���� �ּ��� ������ ã�ƾ� �Ѵ�!
	if (it != paragraph.begin() && util::StringUtil::IsLangChar(start, last, true))
	{
		_paragraph::iterator start = it - 1;

		while (true)
		{
			// �տ� ���ڳ� ���ڰ� �ƴѰ� �߰ߵǾ���. ���⿡�� it������ �� ������!
			if (!util::StringUtil::IsLangChar(start->c_str(), NULL, true))
				break;

			if (start == paragraph.begin())
				break;

			--start;
		}
		// ��� ������ ���ڴ� ���ڳ� ���ڿ��� �Ѵ�!
		const char* last_ch = start->c_str() + start->size() - 1;
		if (!util::StringUtil::IsLangChar(last_ch, last_ch + 1, true))
			++start;

		_sentence& start_mail = *start;

		_paragraph::iterator end = it + 1;
		for (_paragraph::iterator next = start + 1; next != end; next++)
		{
			start_mail.append(".");
			start_mail.append(*next);
		}

		paragraph.erase(start + 1, end);

		it = start;
	}
	return it;
}

inline _paragraph::iterator TextContentParser::ArrangeSentencesEndURL(_paragraph& paragraph, _paragraph::iterator it, const char* start)
{
	// �ڿ� ���ڳ� ���ڰ� �ƴѰ� �ϳ��� �����ٸ�, �� ���忡�� ���� �ּ��� ������ ã�ƾ� �Ѵ�!
	if (paragraph.end() - it > 1 && util::StringUtil::IsLangChar(start, NULL, true))
	{
		_paragraph::iterator end = it + 1;
		for (; end != paragraph.end(); end++)
		{
			if (!util::StringUtil::IsLangChar(end->c_str(), NULL, true))
				break;
		}
		// ���� �߰ߵ� ���ڳ� ���ڰ� �ƴѰ� ���Ե� �͵� ���Խ��Ѿ� �Ѵ�.
		// �ֳ��ϸ� �̰͵� �տ� .�� �����ϱ� �����̴�!
		// ex) aaa.bbb.cc dd.ee
		// "aaa" "bbb" "cc dd" "ee" �� �Ǳ� ������. ���� �ƴѰ� �� ���Խ��Ѿ�
		// "aaa.bbb.cc dd" "ee" �� �ȴ�.

		if (end != paragraph.end())
		{
			// ��� ù��° ���ڴ� ���ڳ� ���ڿ��� �Ѵ�!
			if (util::StringUtil::IsLangChar(end->c_str(), end->c_str() + 1, true))
				++end;
		}

		// �տ� ���ڳ� ���ڰ� �ƴѰ� �߰ߵǾ���. ���⿡�� it������ �� ������!

		_sentence& start_mail = *it;

		for (_paragraph::iterator next = it + 1; next != end; next++)
		{
			start_mail.append(".");
			start_mail.append(*next);
		}

		paragraph.erase(it + 1, end);
	}
	return it;
}
