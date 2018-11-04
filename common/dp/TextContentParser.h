#pragma once

namespace np
{
	namespace ndp
	{
		class TextContentParser
		{
		public:
			static void ParseContent(const std::string& paragraph_token, const std::vector<std::string>& text_vector, _content& content);
			static void ParseText(const std::string& paragraph_token, const std::string& str, _content_col& text);
			static void ParseParagraph(const std::string& str, _paragraph& paragraph);

#ifdef _DEBUG
		public:
#endif
			static void AddSentence(_paragraph& paragraph, const char* start, const char* last);
			static void ArrangeSentenceVector(_paragraph& paragraph);
			static _paragraph::iterator ArrangeSentenceVectorForEmail(_paragraph& paragraph, _paragraph::iterator it);
			static _paragraph::iterator ArrangeSentenceVectorForWeb(_paragraph& paragraph, _paragraph::iterator it);

			static _paragraph::iterator ArrangeSentencesStartURL(_paragraph& paragraph, _paragraph::iterator it, const char* start, const char* last);
			static _paragraph::iterator ArrangeSentencesEndURL(_paragraph& paragraph, _paragraph::iterator it, const char* start);
		;
	}
}
