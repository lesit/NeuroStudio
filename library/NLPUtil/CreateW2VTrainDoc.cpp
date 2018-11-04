#include "CreateW2VTrainDoc.h"

#include "dp/TextParsingReader.h"
#include "util/np_util.h"
#include "util/FileUtil.h"

#include "MecabParser.h"
#include <random>

using namespace np::dp;
using namespace np::nlp;

CreateW2VTrainDoc::CreateW2VTrainDoc()
{
}


CreateW2VTrainDoc::~CreateW2VTrainDoc()
{
}


bool CreateW2VTrainDoc::Create(const char* path, bool hasHeader, bool skip_firstline
	, bool transform_to_fastText
	, size_t split_axis, bool shuffle
	, int setup_max_words, int setup_max_sentences, int setup_max_words_per_sentence
	, std::string outfile_path, size_t flush_count, recv_signal* signal)
{
	_CONTENT_DELIMITER token;
	if (transform_to_fastText)
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));

		skip_firstline = hasHeader;

		setup_max_words = 0;
		setup_max_sentences = 0;
		setup_max_words_per_sentence = 0;
	}
	else
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER());
	}

	token.change_vector.push_back({ "\"\"", "\'" });
	token.change_vector.push_back({ "\r", "" });

	TextParsingReader parser(&token);
	if (!parser.SetFile(path))
		return false;

	MecabParser mecab_parser("./mecabrc");
	if (!mecab_parser.Initialize())
		return false;

	SentenceToWord default_parser;
	if (!default_parser.Initialize())
		return false;

	std::string content_token = token.token;

	_recv_status recv_status;
	memset(&recv_status, 0, sizeof(_recv_status));
	np::Timer timer;
	timer.restart();

	size_t total_content_count = parser.CalculateContentCount();
	size_t start_index = skip_firstline || !transform_to_fastText && hasHeader ? 1 : 0;	// skip하지 않더라도 데이터를 만드려고 한다면 헤더는 스킵하자

	std::vector<std::vector<neuro_size_t>> index_vector_vector;
	index_vector_vector.resize(split_axis > 0 ? 2 : 1);
	if (split_axis > 0)
	{
		size_t middel_split1 = start_index + split_axis / 2;
		size_t middel_index = start_index + (total_content_count - start_index) / 2;
		size_t middel_split2 = middel_index + split_axis / 2;

		std::vector<neuro_size_t>& index_vector1 = index_vector_vector[0];
		index_vector1.resize(split_axis);
		size_t index = 0;
		for (size_t pos = start_index; pos<middel_split1; pos++)
			index_vector1[index++] = pos;

		for (size_t pos = middel_index; pos<middel_split2; pos++)
			index_vector1[index++] = pos;

		std::vector<neuro_size_t>& index_vector2 = index_vector_vector[1];
		index_vector2.resize(total_content_count - start_index - index_vector1.size());
		index = 0;
		for (size_t pos = middel_split1; pos < middel_index; pos++)
			index_vector2[index++] = pos;

		for (size_t pos = middel_split2; pos < total_content_count; pos++)
			index_vector2[index++] = pos;
	}
	else
	{
		std::vector<neuro_size_t>& index_vector = index_vector_vector[0];
		index_vector.resize(total_content_count - start_index);
		size_t index = 0;
		for (size_t pos = start_index; pos < total_content_count; pos++)
			index_vector[index++] = pos;
	}

	std::wstring w_outfile_name = util::StringUtil::MultiByteToWide(util::FileUtil::GetNameFromFileName<char>(outfile_path));
	std::wstring w_outfile_ext = util::StringUtil::MultiByteToWide(util::FileUtil::GetExtFromFileName<char>(outfile_path));
	for (int split = 0; split< index_vector_vector.size(); split++)
	{
		std::wstring w_outfile_path = w_outfile_name;
		if (transform_to_fastText)
		{
			w_outfile_path.append(L"_fasttext.txt");
		}
		else
		{
			w_outfile_path.append(split_axis == 0 ? L"_all_train" : (split == 0 ? L"_train" : L"_test"));
			w_outfile_path.append(w_outfile_ext);
		}

		HANDLE file_handle = INVALID_HANDLE_VALUE;
		DWORD byteswritten;

		file_handle = CreateFile(w_outfile_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
		if (file_handle == INVALID_HANDLE_VALUE)
			return false;

		if (!transform_to_fastText)
		{
			const unsigned char bom[] = { 0xEF, 0xBB, 0xBF };
			WriteFile(file_handle, bom, _countof(bom), &byteswritten, NULL);
		}

		std::vector<neuro_size_t>& index_vector = index_vector_vector[split];
		if (shuffle && !transform_to_fastText)
		{
			std::mt19937 rd_gen;
			std::shuffle(index_vector.begin(), index_vector.end(), rd_gen);
		}

		size_t max_sentence_in_content = 0;
		size_t max_word_in_content = 0;
		size_t sentence_max_word_in_content = 0;
		std::vector<size_t> max_sentence_in_text;
		std::vector<size_t> max_word_in_text;
		size_t max_word_in_sentence = 0;

		neuro_u32 max_sentences_skip_string_count = 0;
		neuro_u32 max_total_words_skip_string_count = 0;
		neuro_u32 max_words_skip_string_count = 0;
		std::string max_sentences_over_skip_strings;
		std::string max_total_words_over_skip_strings;
		std::string max_words_over_skip_strings;

		bool eof;
		// train용 데이터를 만드는 중이고 스킵하지 않았는데 헤더를 가졌으면
		if (!transform_to_fastText && !skip_firstline && hasHeader)
		{
			std::vector<std::string> header_text_vector;
			parser.MoveContentPosition(0);
			if (!parser.ReadContent(NULL, &header_text_vector, eof))
				return false;
			if (header_text_vector.size() == 0)
				return false;

			std::string content_str;
			for (int text_i = 0; text_i < header_text_vector.size(); text_i++)
			{
				if (text_i >0)	// 두번째 text(즉, 두번째 항목)
					content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

				content_str += "\"";
				content_str.append(header_text_vector[text_i]);
				content_str += "\"";
			}
			content_str.append(content_token);

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);

			DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(content_str).append(L"\r\n").c_str());
		}

		size_t cur_total_content = 0;

		// train과 test 파일 두개로 만들고 첫번째 train 파일을 만들고 있으면 split_axis 보다 작아야 한다.
		for (size_t index = 0, n = index_vector.size(); index < n; index++)
		{
#ifdef _DEBUG
			if (index == 500)
				int a = 0;
#endif
			if (!parser.MoveContentPosition(index_vector[index]))
				return false;

			std::vector<std::string> text_vector;
			if (!parser.ReadContent(NULL, &text_vector, eof))
				return false;

			if (text_vector.size() == 0)
				continue;

			size_t prev=max_sentence_in_text.size();
			max_sentence_in_text.resize(text_vector.size());
			for (; prev < max_sentence_in_text.size(); prev++)
				max_sentence_in_text[prev] = 0;

			prev = max_word_in_text.size();
			max_word_in_text.resize(text_vector.size());
			for (; prev < max_word_in_text.size(); prev++)
				max_word_in_text[prev] = 0;

			size_t sentence_in_content = 0;
			size_t word_in_content = 0;

			size_t over_max_sentences_in_content = 0;
			size_t over_max_words_in_content = 0;
			size_t over_max_words_in_sentence = 0;

			std::string content_str;
			for (int text_i = 0; text_i < text_vector.size(); text_i++)
			{
				const std::string& text = text_vector[text_i];

				if (text.size() == 0)
					continue;

				size_t sentence_in_text = 0;
				size_t word_in_text = 0;

				const char* paragraph = text.c_str();
				const char* text_last = text.c_str() + text.size();
				while (paragraph < text_last)
				{
					// 그냥 통째로 mecab에 넣었더니 잘 안된다.
					const char* paragraph_end = strchr(paragraph, '\n');	
					if (paragraph == paragraph_end)
					{
						++paragraph;
						continue;
					}
					if (paragraph_end == NULL)
						paragraph_end = text_last;

					std::vector<_pair_string_vector> sentence_vector;
					if (!mecab_parser.ParseText(paragraph, paragraph_end, sentence_vector))
					{
						DEBUG_OUTPUT(L"failed parse by mecab");
						return false;
					}

					paragraph = paragraph_end + 1;

					for (int sent_i = 0; sent_i < sentence_vector.size(); sent_i++)
					{
						_pair_string_vector& word_vector = sentence_vector[sent_i];

						if (setup_max_words_per_sentence > 0 && setup_max_words_per_sentence<word_vector.size())
						{
							over_max_words_in_sentence = word_vector.size();
							break;
						}

						if (setup_max_words>0 && setup_max_words<word_in_content + word_in_text + word_vector.size())
						{
							over_max_words_in_content = word_in_content + word_in_text + word_vector.size();
							break;
						}
						if (word_vector.size()>0)
						{
							word_in_text += word_vector.size();
							if (max_word_in_sentence < word_vector.size())
								max_word_in_sentence = word_vector.size();

							if (transform_to_fastText)
							{
								content_str.append(word_vector[0].first);
								for (size_t i = 1, n = word_vector.size(); i < n; i++)
								{
									content_str.append(" ");
									content_str.append(word_vector[i].first);
								}
								// fastText를 위해 문장을 구분해줘야 하기 때문에 \n를 넣어준다.
								content_str.append("\n");
							}
						}
					}

					if (over_max_words_in_sentence > 0)
						break;

					if (over_max_words_in_content > 0)
						break;

					sentence_in_text += sentence_vector.size();

					if (setup_max_sentences > 0 && setup_max_sentences < sentence_in_content + sentence_in_text)
					{
						over_max_sentences_in_content = sentence_in_content + sentence_in_text;
						break;
					}
				}

				if (over_max_words_in_sentence>0)
					break;

				if (over_max_words_in_content>0)
					break;

				if (over_max_sentences_in_content > 0)
					break;

				if (max_sentence_in_text[text_i] < sentence_in_text)
					max_sentence_in_text[text_i] = sentence_in_text;

				if (max_word_in_text[text_i] < word_in_text)
					max_word_in_text[text_i] = word_in_text;

				sentence_in_content += sentence_in_text;
				word_in_content += word_in_text;
			}

			if (over_max_sentences_in_content>0)
			{
				//			DEBUG_OUTPUT(L"sentence count[%d] is over. text title is[%s]", sentence_in_content, text_vector[0].c_str());

				++max_sentences_skip_string_count;
				max_sentences_over_skip_strings.append(text_vector[0]);
				max_sentences_over_skip_strings.append("\r\n");
				continue;
			}

			if (over_max_words_in_content > 0)
			{
				++max_total_words_skip_string_count;
				max_total_words_over_skip_strings.append(text_vector[0]);
				max_total_words_over_skip_strings.append("\r\n");
				continue;
			}
			if (over_max_words_in_sentence > 0)
			{
				//			DEBUG_OUTPUT(L"paragraph count[%d] is over. text title is[%s]", paragraph_in_content, text_vector[0].c_str());

				++max_words_skip_string_count;
				max_words_over_skip_strings.append(text_vector[0]);
				max_words_over_skip_strings.append("\r\n");
				continue;
			}

			if (max_sentence_in_content < sentence_in_content)
				max_sentence_in_content = sentence_in_content;
			if (max_word_in_content < word_in_content)
			{
				max_word_in_content = word_in_content;
				sentence_max_word_in_content = sentence_in_content;	// 최대 워드수일때의 실제 문장수를 체크해보자
			}

			recv_status.total_sentence += sentence_in_content;
			recv_status.total_word += word_in_content;

			if (!transform_to_fastText)
			{
				for (int text_i = 0; text_i < text_vector.size(); text_i++)
				{
					if (text_i >0)	// 두번째 text(즉, 두번째 항목)
						content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

					content_str += "\"";
					content_str.append(text_vector[text_i]);
					content_str += "\"";
				}
				content_str.append(content_token);
			}

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);

			++recv_status.total_content;

			++cur_total_content;
			if (cur_total_content % flush_count == 0)
			{
				if (file_handle != INVALID_HANDLE_VALUE)
					FlushFileBuffers(file_handle);

				recv_status.elapse = timer.elapsed();
				if (signal)
					signal->signal(recv_status);
			}
		}
		if (file_handle != INVALID_HANDLE_VALUE)
			CloseHandle(file_handle);

		recv_status.elapse = timer.elapsed();

		{
			std::wstring analysis_file = w_outfile_path;
			analysis_file.append(L".ini");
			WritePrivateProfileString(L"setup", L"skip first line", skip_firstline ? L"true" : L"false", analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in content", util::StringUtil::Transform<wchar_t>(setup_max_words).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max sentences in content", util::StringUtil::Transform<wchar_t>(setup_max_sentences).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in sentence", util::StringUtil::Transform<wchar_t>(setup_max_words_per_sentence).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"total", L"content", util::StringUtil::Transform<wchar_t>(cur_total_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"sentence", util::StringUtil::Transform<wchar_t>(recv_status.total_sentence).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"word", util::StringUtil::Transform<wchar_t>(recv_status.total_word).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"elapse", util::StringUtil::Transform<wchar_t>(recv_status.elapse).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"content", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"sentence when max word", util::StringUtil::Transform<wchar_t>(sentence_max_word_in_content).c_str(), analysis_file.c_str());

			for (size_t i = 0; i < max_sentence_in_text.size(); i++)
			{
				std::wstring key;
				key = L"max sentence"; key.append(util::StringUtil::Transform<wchar_t>(i));
				WritePrivateProfileString(L"text", key.c_str(), util::StringUtil::Transform<wchar_t>(max_sentence_in_text[i]).c_str(), analysis_file.c_str());

				key = L"max word"; key.append(util::StringUtil::Transform<wchar_t>(i));
				WritePrivateProfileString(L"text", key.c_str(), util::StringUtil::Transform<wchar_t>(max_word_in_text[i]).c_str(), analysis_file.c_str());
			}

			WritePrivateProfileString(L"sentence", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_sentence).c_str(), analysis_file.c_str());
		}

		std::wstring skip_strings_fname = w_outfile_path;
		skip_strings_fname.append(L"_skips.txt");
		if (max_sentences_skip_string_count > 0 || max_total_words_skip_string_count > 0 || max_words_skip_string_count > 0)
		{
			HANDLE skip_file_handle = CreateFile(skip_strings_fname.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
			if (skip_file_handle != INVALID_HANDLE_VALUE)
			{
				std::string str;
				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over total words count=%u\r\n", max_total_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_total_words_over_skip_strings.c_str(), max_total_words_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over sentences count=%u\r\n", max_sentences_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_sentences_over_skip_strings.c_str(), max_sentences_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over words count=%u\r\n", max_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_words_over_skip_strings.c_str(), max_words_over_skip_strings.size(), &byteswritten, NULL);

				CloseHandle(skip_file_handle);
			}
		}
		else
		{
			DeleteFile(skip_strings_fname.c_str());
		}

		if (signal)
			signal->signal(recv_status);
	}
	return true;
}

/*
bool CreateW2VTrainDoc::Create(const char* path, bool hasHeader, bool skip_firstline
	, bool transform_to_fastText
	, int setup_max_words, int setup_max_sentences, int setup_max_words_per_sentence
	, std::string outfile_path, size_t split_axis, size_t flush_count, recv_signal* signal)
{
	_CONTENT_DELIMITER token;
	if (transform_to_fastText)
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));

		skip_firstline = hasHeader;

		setup_max_words = 0;
		setup_max_sentences = 0;
		setup_max_words_per_sentence = 0;
	}
	else
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER());
	}

	token.change_vector.push_back({ "\"\"", "\'" });
	token.change_vector.push_back({ "\r", "" });

	TextParsingReader parser(&token);
	if (!parser.SetFile(path))
		return false;

	MecabParser mecab_parser("./mecabrc");
	if (!mecab_parser.Initialize())
		return false;

	SentenceToWord default_parser;
	if (!default_parser.Initialize())
		return false;

	std::string content_token = token.token;

	_recv_status recv_status;
	memset(&recv_status, 0, sizeof(_recv_status));
	np::Timer timer;
	timer.restart();

	size_t total_content_count = parser.CalculateContentCount();
	size_t start_index = skip_firstline || !transform_to_fastText && hasHeader ? 1 : 0;	// skip하지 않더라도 데이터를 만드려고 한다면 헤더는 스킵하자

	std::vector<std::vector<neuro_size_t>> index_vector_vector;
	index_vector_vector.resize(split_axis > 0 ? 2 : 1);
	if (split_axis > 0)
	{
		size_t middel_split1 = start_index + split_axis / 2;
		size_t middel_index = start_index + (total_content_count - start_index) / 2;
		size_t middel_split2 = middel_index + split_axis / 2;

		std::vector<neuro_size_t>& index_vector1 = index_vector_vector[0];
		index_vector1.resize(split_axis);
		size_t index = 0;
		for (size_t pos = start_index; pos<middel_split1; pos++)
			index_vector1[index++] = pos;

		for (size_t pos = middel_index; pos<middel_split2; pos++)
			index_vector1[index++] = pos;

		std::vector<neuro_size_t>& index_vector2 = index_vector_vector[1];
		index_vector2.resize(total_content_count - start_index - index_vector1.size());
		index = 0;
		for (size_t pos = middel_split1; pos < middel_index; pos++)
			index_vector2[index++] = pos;

		for (size_t pos = middel_split2; pos < total_content_count; pos++)
			index_vector2[index++] = pos;
	}
	else
	{
		std::vector<neuro_size_t>& index_vector = index_vector_vector[0];
		index_vector.resize(total_content_count - start_index);
		size_t index = 0;
		for (size_t pos = start_index; pos < total_content_count; pos++)
			index_vector[index] = pos;
	}

	bool failed = false;

	std::wstring w_outfile_name = util::StringUtil::MultiByteToWide(util::FileUtil::GetNameFromFileName<char>(outfile_path));
	std::wstring w_outfile_ext = util::StringUtil::MultiByteToWide(util::FileUtil::GetExtFromFileName<char>(outfile_path));
	for (int split = 0; split< index_vector_vector.size(); split++)
	{
		std::wstring w_outfile_path = w_outfile_name;
		if (transform_to_fastText)
		{
			w_outfile_path.append(L"_fasttext.txt");
		}
		else
		{
			w_outfile_path.append(split_axis == 0 ? L"_all_train" : (split == 0 ? L"_train" : L"_test"));
			w_outfile_path.append(w_outfile_ext);
		}

		HANDLE file_handle = INVALID_HANDLE_VALUE;
		DWORD byteswritten;

		file_handle = CreateFile(w_outfile_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
		if (file_handle == INVALID_HANDLE_VALUE)
			return false;

		if (!transform_to_fastText)
		{
			const unsigned char bom[] = { 0xEF, 0xBB, 0xBF };
			WriteFile(file_handle, bom, _countof(bom), &byteswritten, NULL);
		}

		std::vector<neuro_size_t>& index_vector = index_vector_vector[split];
		if (!transform_to_fastText)
		{
			std::mt19937 rd_gen;
			std::shuffle(index_vector.begin(), index_vector.end(), rd_gen);
		}

		size_t max_paragraph_in_content = 0;
		size_t max_sentence_in_content = 0;
		size_t max_word_in_content = 0;
		size_t sentence_max_word_in_content = 0;
		size_t max_paragraph_in_text = 0;
		size_t max_sentence_in_text = 0;
		size_t max_word_in_text = 0;
		size_t max_sentence_in_paragraph = 0;
		size_t max_word_in_paragraph = 0;
		size_t max_word_in_sentence = 0;

		neuro_u32 max_sentences_skip_string_count = 0;
		neuro_u32 max_total_words_skip_string_count = 0;
		neuro_u32 max_words_skip_string_count = 0;
		std::string max_sentences_over_skip_strings;
		std::string max_total_words_over_skip_strings;
		std::string max_words_over_skip_strings;

		bool eof;
		// train용 데이터를 만드는 중이고 스킵하지 않았는데 헤더를 가졌으면
		if (!transform_to_fastText && !skip_firstline && hasHeader)
		{
			std::vector<std::string> header_text_vector;
			parser.MoveContentPosition(0);
			if (!parser.ReadContent(NULL, &header_text_vector, eof))
				return false;
			if (header_text_vector.size() == 0)
				return false;

			std::string content_str;
			for (int text_i = 0; text_i < header_text_vector.size(); text_i++)
			{
				if (text_i >0)	// 두번째 text(즉, 두번째 항목)
					content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

				content_str += "\"";
				content_str.append(header_text_vector[text_i]);
				content_str += "\"";
			}
			content_str.append(content_token);

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);

			DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(content_str).append(L"\r\n").c_str());
		}

		size_t cur_total_content = 0;

		// train과 test 파일 두개로 만들고 첫번째 train 파일을 만들고 있으면 split_axis 보다 작아야 한다.
		for (size_t index=0, n = index_vector.size(); index < n; index++)
		{
#ifdef _DEBUG
			if (index == 500)
				int a = 0;
#endif
			if(!parser.MoveContentPosition(index_vector[index]))
				return false;

			std::vector<std::string> text_vector;
			if (!parser.ReadContent(NULL, &text_vector, eof))
				return false;

			if (text_vector.size() == 0)
				continue;

			size_t paragraph_in_content = 0;
			size_t sentence_in_content = 0;
			size_t word_in_content = 0;

			size_t over_max_sentences_in_content = 0;
			size_t over_max_words_in_content = 0;
			size_t over_max_words_in_sentence = 0;

			std::string content_str;
			for (int text_i=0; text_i < text_vector.size(); text_i++)
			{
				const std::string& text = text_vector[text_i];

				size_t sentence_in_text = 0;
				size_t word_in_text = 0;

				const char* paragraph = text.c_str();
				const char* text_last = text.c_str() + text.size();
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

					std::vector<_pair_string_vector> sentence_vector;
					if (!mecab_parser.ParseText(paragraph, paragraph_end, sentence_vector))
					{
						DEBUG_OUTPUT(L"failed parse by mecab");
						failed = true;
					}

					size_t word_in_paragraph = 0;

					for (int sent_i = 0; sent_i < sentence_vector.size(); sent_i++)
					{
						_pair_string_vector& word_vector = sentence_vector[sent_i];

						if (setup_max_words_per_sentence > 0 && setup_max_words_per_sentence<word_vector.size())
						{
							over_max_words_in_sentence = word_vector.size();
							break;
						}

						if (setup_max_words>0 && setup_max_words<word_in_content + word_in_text + word_vector.size())
						{
							over_max_words_in_content = word_in_content + word_in_text + word_vector.size();
							break;
						}
						if (word_vector.size()>0)
						{
							word_in_text += word_vector.size();
							word_in_paragraph += word_vector.size();
							if (max_word_in_sentence < word_vector.size())
								max_word_in_sentence = word_vector.size();

							if (transform_to_fastText)
							{
								content_str.append(word_vector[0].first);
								for (size_t i = 1, n = word_vector.size(); i < n; i++)
								{
									content_str.append(" ");
									content_str.append(word_vector[i].first);
								}
								// fastText를 위해 문장을 구분해줘야 하기 때문에 \n를 넣어준다.
								content_str.append("\n");
							}
						}
					}
					if (over_max_words_in_sentence>0)
						break;

					if (over_max_words_in_content > 0)
						break;

					sentence_in_text += sentence_vector.size();

					if (setup_max_sentences > 0 && setup_max_sentences < sentence_in_content + sentence_in_text)
					{
						over_max_sentences_in_content = sentence_in_content + sentence_in_text;
						break;
					}

					if (max_sentence_in_paragraph < sentence_vector.size())
						max_sentence_in_paragraph = sentence_vector.size();

					if (max_word_in_paragraph < word_in_paragraph)
						max_word_in_paragraph = word_in_paragraph;

					paragraph = paragraph_end+1;
				}
				if (failed)
					break;

				if (over_max_words_in_sentence>0)
					break;

				if (over_max_words_in_content>0)
					break;

				if (over_max_sentences_in_content > 0)
					break;

				if (max_paragraph_in_text < text.size())
					max_paragraph_in_text = text.size();

				if (max_sentence_in_text < sentence_in_text)
					max_sentence_in_text = sentence_in_text;

				if (max_word_in_text < word_in_text)
					max_word_in_text = word_in_text;

				paragraph_in_content += text.size();
				sentence_in_content += sentence_in_text;
				word_in_content += word_in_text;
			}

			if (over_max_sentences_in_content>0)
			{
				//			DEBUG_OUTPUT(L"sentence count[%d] is over. text title is[%s]", sentence_in_content, text_vector[0].c_str());

				++max_sentences_skip_string_count;
				max_sentences_over_skip_strings.append(text_vector[0]);
				max_sentences_over_skip_strings.append("\r\n");
				continue;
			}

			if (over_max_words_in_content > 0)
			{
				++max_total_words_skip_string_count;
				max_total_words_over_skip_strings.append(text_vector[0]);
				max_total_words_over_skip_strings.append("\r\n");
				continue;
			}
			if (over_max_words_in_sentence > 0)
			{
				//			DEBUG_OUTPUT(L"paragraph count[%d] is over. text title is[%s]", paragraph_in_content, text_vector[0].c_str());

				++max_words_skip_string_count;
				max_words_over_skip_strings.append(text_vector[0]);
				max_words_over_skip_strings.append("\r\n");
				continue;
			}

			if (max_paragraph_in_content < paragraph_in_content)
				max_paragraph_in_content = paragraph_in_content;
			if (max_sentence_in_content < sentence_in_content)
				max_sentence_in_content = sentence_in_content;
			if (max_word_in_content < word_in_content)
			{
				max_word_in_content = word_in_content;
				sentence_max_word_in_content = sentence_in_content;	// 최대 워드수일때의 실제 문장수를 체크해보자
			}

			recv_status.total_paragraph += paragraph_in_content;
			recv_status.total_sentence += sentence_in_content;
			recv_status.total_word += word_in_content;

			if (!transform_to_fastText)
			{
				for (int text_i = 0; text_i < text_vector.size(); text_i++)
				{
					if (text_i >0)	// 두번째 text(즉, 두번째 항목)
						content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

					content_str += "\"";
					content_str.append(text_vector[text_i]);
					content_str += "\"";
				}
				content_str.append(content_token);
			}

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);

			++recv_status.total_content;

			++cur_total_content;
			if (cur_total_content % flush_count == 0)
			{
				if (file_handle != INVALID_HANDLE_VALUE)
					FlushFileBuffers(file_handle);

				recv_status.elapse = timer.elapsed();
				if (signal)
					signal->signal(recv_status);
			}
		}
		if (file_handle != INVALID_HANDLE_VALUE)
			CloseHandle(file_handle);

		recv_status.elapse = timer.elapsed();

		{
			std::wstring analysis_file = w_outfile_path;
			analysis_file.append(L".ini");
			WritePrivateProfileString(L"setup", L"skip first line", skip_firstline ? L"true" : L"false", analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in content", util::StringUtil::Transform<wchar_t>(setup_max_words).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max sentences in content", util::StringUtil::Transform<wchar_t>(setup_max_sentences).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in sentence", util::StringUtil::Transform<wchar_t>(setup_max_words_per_sentence).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"total", L"content", util::StringUtil::Transform<wchar_t>(cur_total_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"paragraph", util::StringUtil::Transform<wchar_t>(recv_status.total_paragraph).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"sentence", util::StringUtil::Transform<wchar_t>(recv_status.total_sentence).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"word", util::StringUtil::Transform<wchar_t>(recv_status.total_word).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"elapse", util::StringUtil::Transform<wchar_t>(recv_status.elapse).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"content", L"max paragraph", util::StringUtil::Transform<wchar_t>(max_paragraph_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"sentence when max word", util::StringUtil::Transform<wchar_t>(sentence_max_word_in_content).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"text", L"max paragraph", util::StringUtil::Transform<wchar_t>(max_paragraph_in_text).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"text", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_text).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"text", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_text).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"paragraph", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_paragraph).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"paragraph", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_paragraph).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"sentence", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_sentence).c_str(), analysis_file.c_str());
		}

		std::wstring skip_strings_fname = w_outfile_path;
		skip_strings_fname.append(L"_skips.txt");
		if (max_sentences_skip_string_count > 0 || max_total_words_skip_string_count > 0 || max_words_skip_string_count > 0)
		{
			HANDLE skip_file_handle = CreateFile(skip_strings_fname.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
			if (skip_file_handle != INVALID_HANDLE_VALUE)
			{
				std::string str;
				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over total words count=%u\r\n", max_total_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_total_words_over_skip_strings.c_str(), max_total_words_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over sentences count=%u\r\n", max_sentences_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_sentences_over_skip_strings.c_str(), max_sentences_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over words count=%u\r\n", max_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_words_over_skip_strings.c_str(), max_words_over_skip_strings.size(), &byteswritten, NULL);

				CloseHandle(skip_file_handle);
			}
		}
		else
		{
			DeleteFile(skip_strings_fname.c_str());
		}

		if (signal)
			signal->signal(recv_status);
	}
	return !failed;
}
*/

/*
bool CreateW2VTrainDoc::Create(const char* path, bool hasHeader, bool skip_firstline
	, bool transform_to_fastText
	, int setup_max_words, int setup_max_sentences, int setup_max_words_per_sentence
	, std::string outfile_path, size_t split_axis, size_t flush_count, recv_signal* signal)
{
	_CONTENT_DELIMITER token;
	if (transform_to_fastText)
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }, false));
		
		skip_firstline = hasHeader;

		setup_max_words = 0;
		setup_max_sentences = 0;
		setup_max_words_per_sentence = 0;
	}
	else
	{
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER({ ",", "\t" }));
		token.col_delimiter_vector.push_back(_CONTENT_COL_DELIMITER());
	}

	token.change_vector.push_back({ "\"\"", "\'" });
	token.change_vector.push_back({ "\r", "" });

	TextParsingReader parser(&token);
	if (!parser.SetFile(path))
		return false;

	MecabParser mecab_parser("./mecabrc");
	if (!mecab_parser.Initialize())
		return false;

	SentenceToWord default_parser;
	if (!default_parser.Initialize())
		return false;

	std::string content_token = token.token;

	_recv_status recv_status;
	memset(&recv_status, 0, sizeof(_recv_status));
	np::Timer timer;
	timer.restart();
	
	size_t max_paragraph_in_content = 0;
	size_t max_sentence_in_content = 0;
	size_t max_word_in_content = 0;
	size_t max_paragraph_in_text = 0;
	size_t max_sentence_in_text = 0;
	size_t max_word_in_text = 0;
	size_t max_sentence_in_paragraph = 0;
	size_t max_word_in_paragraph = 0;
	size_t max_word_in_sentence = 0;

	neuro_u32 max_sentences_skip_string_count = 0;
	neuro_u32 max_total_words_skip_string_count = 0;
	neuro_u32 max_words_skip_string_count = 0;
	std::string max_sentences_over_skip_strings;
	std::string max_total_words_over_skip_strings;
	std::string max_words_over_skip_strings;

	bool failed = false;

	bool eof = false;
	size_t index = 0;
	if (skip_firstline)
	{
		++index;
		std::vector<std::string> text_vector;
		if (!parser.ReadContent(NULL, &text_vector, eof))
			return false;
	}

	std::vector<std::string> header_text_vector;

	std::wstring w_outfile_name = util::StringUtil::MultiByteToWide(util::FileUtil::GetNameFromFileName<char>(outfile_path));
	std::wstring w_outfile_ext = util::StringUtil::MultiByteToWide(util::FileUtil::GetExtFromFileName<char>(outfile_path));
	for (int split = 0; split< (split_axis>0 ? 2 : 1) && !eof; split++)
	{
		std::wstring w_outfile_path = w_outfile_name;
		if (transform_to_fastText)
		{
			w_outfile_path.append(L"_fasttext.txt");
		}
		else
		{
			w_outfile_path.append(split_axis == 0 ? L"_all_train" : (split == 0 ? L"_train" : L"_test"));
			w_outfile_path.append(w_outfile_ext);
		}

		HANDLE file_handle = INVALID_HANDLE_VALUE;
		DWORD byteswritten;

		file_handle = CreateFile(w_outfile_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
		if (file_handle == INVALID_HANDLE_VALUE)
			return false;

		if (!transform_to_fastText)
		{
			const unsigned char bom[] = { 0xEF, 0xBB, 0xBF };
			WriteFile(file_handle, bom, _countof(bom), &byteswritten, NULL);
		}

		// train용 데이터를 만드는 중이고 스킵하지 않았는데 헤더를 가졌으면
		if (!transform_to_fastText && !skip_firstline && hasHeader)
		{
			if (split == 0)	// train 용이면 헤더를 저장해 놓자
			{
				if (!parser.ReadContent(NULL, &header_text_vector, eof))
					return false;
				if (header_text_vector.size() == 0)
					return false;
			}

			std::string content_str;
			for (int text_i = 0; text_i < header_text_vector.size(); text_i++)
			{
				if (text_i >0)	// 두번째 text(즉, 두번째 항목)
					content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

				content_str += "\"";
				content_str.append(header_text_vector[text_i]);
				content_str += "\"";
			}
			content_str.append(content_token);

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);
		}

		size_t cur_total_content = 0;

		// train과 test 파일 두개로 만들고 첫번째 train 파일을 만들고 있으면 split_axis 보다 작아야 한다.
		for (; (split_axis == 0 || split>0 || split == 0 && recv_status.total_content < split_axis) && !eof; index++)
		{
#ifdef _DEBUG
			if (index == 500)
				int a = 0;
#endif

			std::vector<std::string> text_vector;
			if (!parser.ReadContent(NULL, &text_vector, eof))
				return false;

			if (text_vector.size() == 0)
				continue;

			_content content;
			TextParsingReader::ParseContent("\n", text_vector, content);
			if (content.size() == 0)
				continue;

			size_t paragraph_in_content = 0;
			size_t sentence_in_content = 0;
			size_t word_in_content = 0;

			size_t over_max_sentences_in_content = 0;
			size_t over_max_words_in_content = 0;
			size_t over_max_words_in_sentence = 0;

			std::string content_str;

			size_t text_i = 0;
			const size_t text_n = content.size();
			for (; text_i < text_n; text_i++)
			{
#ifdef _DEBUG
				if (text_i == 3)
					int a = 0;
#endif
				const _content_col& text = content[text_i];

				size_t sentence_in_text = 0;
				size_t word_in_text = 0;
				for (size_t para_i = 0, para_n = text.size(); para_i < text.size(); para_i++)
				{
					const _paragraph& paragraph = text[para_i];

					size_t word_in_paragraph = 0;

					const size_t sent_n = paragraph.size();
					for (size_t sent_i = 0; sent_i < sent_n; sent_i++)
					{
						const _sentence& sentence = paragraph[sent_i];

						_pair_string_vector word_vector;

						if (!mecab_parser.ParseSentence(sentence, word_vector))
						{
							DEBUG_OUTPUT(L"failed parse by mecab");
							failed = true;
							break;
						}

						if (setup_max_words_per_sentence > 0 && setup_max_words_per_sentence<word_vector.size())
						{
							over_max_words_in_sentence = word_vector.size();
							break;
						}

						if (setup_max_words>0 && setup_max_words<word_in_content + word_in_text + word_vector.size())
						{
							over_max_words_in_content = word_in_content + word_in_text + word_vector.size();
							break;
						}
						if (word_vector.size()>0)
						{
							word_in_text += word_vector.size();
							word_in_paragraph += word_vector.size();
							if (max_word_in_sentence < word_vector.size())
								max_word_in_sentence = word_vector.size();

							if (transform_to_fastText)
							{
								content_str.append(word_vector[0].first);
								for (size_t i = 1, n = word_vector.size(); i < n; i++)
								{
									content_str.append(" ");
									content_str.append(word_vector[i].first);
								}
								// fastText를 위해 문장을 구분해줘야 하기 때문에 \n를 넣어준다.
								content_str.append("\n");
							}
						}
					}
					if (over_max_words_in_sentence>0)
						break;

					if (over_max_words_in_content > 0)
						break;

					sentence_in_text += sent_n;

					if (setup_max_sentences > 0 && setup_max_sentences < sentence_in_content + sentence_in_text)
					{
						over_max_sentences_in_content = sentence_in_content + sentence_in_text;
						break;
					}

					if (max_sentence_in_paragraph < sent_n)
						max_sentence_in_paragraph = sent_n;

					if (max_word_in_paragraph < word_in_paragraph)
						max_word_in_paragraph = word_in_paragraph;
				}

				if (over_max_words_in_sentence>0)
					break;

				if (over_max_words_in_content>0)
					break;

				if (over_max_sentences_in_content > 0)
					break;

				if (max_paragraph_in_text < text.size())
					max_paragraph_in_text = text.size();

				if (max_sentence_in_text < sentence_in_text)
					max_sentence_in_text = sentence_in_text;

				if (max_word_in_text < word_in_text)
					max_word_in_text = word_in_text;

				paragraph_in_content += text.size();
				sentence_in_content += sentence_in_text;
				word_in_content += word_in_text;
			}
			if (failed)
				break;

			if (over_max_sentences_in_content>0)
			{
				//			DEBUG_OUTPUT(L"sentence count[%d] is over. text title is[%s]", sentence_in_content, text_vector[0].c_str());

				++max_sentences_skip_string_count;
				max_sentences_over_skip_strings.append(text_vector[0]);
				max_sentences_over_skip_strings.append("\r\n");
				continue;
			}

			if (over_max_words_in_content > 0)
			{
				++max_total_words_skip_string_count;
				max_total_words_over_skip_strings.append(text_vector[0]);
				max_total_words_over_skip_strings.append("\r\n");
				continue;
			}
			if (over_max_words_in_sentence > 0)
			{
				//			DEBUG_OUTPUT(L"paragraph count[%d] is over. text title is[%s]", paragraph_in_content, text_vector[0].c_str());

				++max_words_skip_string_count;
				max_words_over_skip_strings.append(text_vector[0]);
				max_words_over_skip_strings.append("\r\n");
				continue;
			}

			if (max_paragraph_in_content < paragraph_in_content)
				max_paragraph_in_content = paragraph_in_content;
			if (max_sentence_in_content < sentence_in_content)
				max_sentence_in_content = sentence_in_content;
			if (max_word_in_content < word_in_content)
				max_word_in_content = word_in_content;

			recv_status.total_paragraph += paragraph_in_content;
			recv_status.total_sentence += sentence_in_content;
			recv_status.total_word += word_in_content;

			if (!transform_to_fastText)
			{
				for (int text_i = 0; text_i < text_vector.size(); text_i++)
				{
					if (text_i >0)	// 두번째 text(즉, 두번째 항목)
						content_str += token.col_delimiter_vector[text_i - 1].token_vector[0];

					content_str += "\"";
					content_str.append(text_vector[text_i]);
					content_str += "\"";
				}
				content_str.append(content_token);
			}

			WriteFile(file_handle, content_str.c_str(), content_str.size(), &byteswritten, NULL);

			++recv_status.total_content;

			++cur_total_content;
			if (cur_total_content % flush_count == 0)
			{
				if (file_handle != INVALID_HANDLE_VALUE)
					FlushFileBuffers(file_handle);

				recv_status.elapse = timer.elapsed();
				if (signal)
					signal->signal(recv_status);
			}
		}
		if (file_handle != INVALID_HANDLE_VALUE)
			CloseHandle(file_handle);

		recv_status.elapse = timer.elapsed();

		{
			std::wstring analysis_file = w_outfile_path;
			analysis_file.append(L".ini");
			WritePrivateProfileString(L"setup", L"skip first line", skip_firstline ? L"true" : L"false", analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in content", util::StringUtil::Transform<wchar_t>(setup_max_words).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max sentences in content", util::StringUtil::Transform<wchar_t>(setup_max_sentences).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"setup", L"max words in sentence", util::StringUtil::Transform<wchar_t>(setup_max_words_per_sentence).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"total", L"content", util::StringUtil::Transform<wchar_t>(cur_total_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"paragraph", util::StringUtil::Transform<wchar_t>(recv_status.total_paragraph).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"sentence", util::StringUtil::Transform<wchar_t>(recv_status.total_sentence).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"word", util::StringUtil::Transform<wchar_t>(recv_status.total_word).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"total", L"elapse", util::StringUtil::Transform<wchar_t>(recv_status.elapse).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"content", L"max paragraph", util::StringUtil::Transform<wchar_t>(max_paragraph_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_content).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"content", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_content).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"text", L"max paragraph", util::StringUtil::Transform<wchar_t>(max_paragraph_in_text).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"text", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_text).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"text", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_text).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"paragraph", L"max sentence", util::StringUtil::Transform<wchar_t>(max_sentence_in_paragraph).c_str(), analysis_file.c_str());
			WritePrivateProfileString(L"paragraph", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_paragraph).c_str(), analysis_file.c_str());

			WritePrivateProfileString(L"sentence", L"max word", util::StringUtil::Transform<wchar_t>(max_word_in_sentence).c_str(), analysis_file.c_str());
		}

		std::wstring skip_strings_fname = w_outfile_path;
		skip_strings_fname.append(L"_skips.txt");
		if (max_sentences_skip_string_count > 0 || max_total_words_skip_string_count > 0 || max_words_skip_string_count > 0)
		{
			HANDLE skip_file_handle = CreateFile(skip_strings_fname.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);;
			if (skip_file_handle != INVALID_HANDLE_VALUE)
			{
				std::string str;
				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over total words count=%u\r\n", max_total_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_total_words_over_skip_strings.c_str(), max_total_words_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over sentences count=%u\r\n", max_sentences_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_sentences_over_skip_strings.c_str(), max_sentences_over_skip_strings.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, "\r\n", 2, &byteswritten, NULL);

				str = util::StringUtil::WideToMultiByte(util::StringUtil::Format<wchar_t>(L"over words count=%u\r\n", max_words_skip_string_count));
				WriteFile(skip_file_handle, str.c_str(), str.size(), &byteswritten, NULL);
				WriteFile(skip_file_handle, max_words_over_skip_strings.c_str(), max_words_over_skip_strings.size(), &byteswritten, NULL);

				CloseHandle(skip_file_handle);
			}
		}
		else
		{
			DeleteFile(skip_strings_fname.c_str());
		}

		if (signal)
			signal->signal(recv_status);
	}
	return !failed;
}
*/