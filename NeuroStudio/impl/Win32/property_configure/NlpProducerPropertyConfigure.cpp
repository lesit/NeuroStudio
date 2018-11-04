#include "stdafx.h"
#include "DataProducerPropertyConfigure.h"

#include "NeuroData/model/NlpProducerModel.h"

using namespace property;

void NlpProducerPropertyConfigure::CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractProducerModel* model)
{
	NlpProducerModel* producer = (NlpProducerModel*)model;

	CMFCPropertyGridProperty* nlp_prop = new CMFCPropertyGridProperty(L"Natural Language");
	list_ctrl.AddProperty(nlp_prop);

	static const TCHAR szFilter[] = L"All Files (*.*)|*.*||";
	{
		CMFCPropertyGridProperty* morpheme_prop = new CMFCPropertyGridProperty(L"Morpheme parser");
		nlp_prop->AddSubItem(morpheme_prop);

		CMFCPropertyGridProperty* use_parse_prop = new CMFCPropertyGridProperty(L"Use morpheme parser"
			, (_variant_t)producer->UseMorphemeParser()
			, L"Whether using or not a morpheme parser"
			, (DWORD_PTR)_prop_type::morpheme_parser);
		morpheme_prop->AddSubItem(use_parse_prop);

		if (producer->UseMorphemeParser())
		{
			CMFCPropertyGridFileProperty* mecap_rc_prop = new CMFCPropertyGridFileProperty(L"Mecab rc", TRUE
				, util::StringUtil::MultiByteToWide(producer->GetMecapRcPath()).c_str()
				, L"", 0, szFilter, L"Resource file for Mecap that is using to parse morphemes"
				, (DWORD_PTR)_prop_type::mecaprc_path);
			morpheme_prop->AddSubItem(mecap_rc_prop);

			CMFCPropertyGridProperty* use_morpheme_type_vector_prop = new CMFCPropertyGridProperty(L"Morpheme type vector"
				, (_variant_t)producer->UseMorphemeTypeVector()
				, L"Vector for morpheme type of each word"
				, (DWORD_PTR)_prop_type::use_morpheme_type_vector);
			morpheme_prop->AddSubItem(use_morpheme_type_vector_prop);
		}
	}

	{
		CMFCPropertyGridProperty* w2v_prop = new CMFCPropertyGridProperty(L"Word embedding");
		nlp_prop->AddSubItem(w2v_prop);

		CMFCPropertyGridFileProperty* w2v_file_prop = new CMFCPropertyGridFileProperty(L"W2V file", TRUE
			, util::StringUtil::MultiByteToWide(producer->GetWordToVectorPath()).c_str()
			, L"", 0, szFilter, L"'Word to Vector' file to word embed"
			, (DWORD_PTR)_prop_type::w2v_path);
		w2v_prop->AddSubItem(w2v_file_prop);

		CMFCPropertyGridProperty* prop = new CMFCPropertyGridProperty(L"Dimension"
			, (_variant_t)producer->GetWordDimension()
			, L"Dimension of word");
		prop->AllowEdit(FALSE);
		w2v_prop->AddSubItem(prop);

		prop = new CMFCPropertyGridProperty(L"Vector normalization"
			, (_variant_t)producer->IsVectorNormalization()
			, L"Normalize values in word vector"
			, (DWORD_PTR)_prop_type::word_norm);
		w2v_prop->AddSubItem(prop);
	}
		
	const _u32_set& index_set = producer->GetUsingSourceColumnSet();

	CModelGridProperty* column_list_prop = new CModelGridProperty(L"Column list");
	list_ctrl.AddProperty(column_list_prop);
	if (producer->GetInput())
	{
		neuro_u32 input_column_count = producer->GetInput()->GetColumnCount();
		for (neuro_u32 i = 0; i < input_column_count; i++)
		{
			bool use = index_set.find(i) != index_set.end();

			CModelGridProperty* column_prop = new CModelGridProperty(util::StringUtil::Format<wchar_t>(L"%u column", i).c_str()
				, (_variant_t)use
				, L"Select using or not", (DWORD_PTR)_prop_type::src_use);
			column_prop->index = i;
			column_list_prop->AddSubItem(column_prop);
		}
	}

	{
		CModelGridProperty* data_shape_prop = new CModelGridProperty(L"Data shape");
		list_ctrl.AddProperty(data_shape_prop);

		CModelGridProperty* parse_sentence_prop = new CModelGridProperty(L"Parse sentence"
			, (_variant_t)producer->ParsingSentence()
			, L"Parse sentence. If you select yes, data has 2D shape and a column is word vector and a row has vector of words in sentence."
			, (DWORD_PTR)_prop_type::parse_sentence);
		data_shape_prop->AddSubItem(parse_sentence_prop);

		if (producer->ParsingSentence())
		{
			CModelGridProperty* prop = new CModelGridProperty(L"Max sentences"
				, (_variant_t)producer->GetMaxSentence()
				, L"a max count of word in text"
				, (DWORD_PTR)_prop_type::max_sentence);
			data_shape_prop->AddSubItem(prop);

			prop = new CModelGridProperty(L"Max words per sentence"
				, (_variant_t)producer->GetMaxWordPerSentence()
				, L"a max count of word in text"
				, (DWORD_PTR)_prop_type::max_sentence);
			data_shape_prop->AddSubItem(prop);
		}
		else
		{
			CModelGridProperty* prop = new CModelGridProperty(L"Max words"
				, (_variant_t)producer->GetMaxWord()
				, L"a max count of word in text"
				, (DWORD_PTR)_prop_type::max_word);
			data_shape_prop->AddSubItem(prop);
		}
	}
}
	
void NlpProducerPropertyConfigure::PropertyChanged(CModelGridProperty* prop, AbstractProducerModel* model, bool& reload) const
{
	NlpProducerModel* producer = (NlpProducerModel*)model;

	_prop_type prop_type = (_prop_type) prop->GetData();
	if (prop_type == _prop_type::morpheme_parser)
	{
		producer->SetUseMorphemeParser(prop->GetValue().boolVal);
		reload = true;
	}
	else if (prop_type == _prop_type::mecaprc_path)
	{
		producer->SetMecapRcPath(CStringA(prop->GetValue()));
	}
	else if (prop_type == _prop_type::use_morpheme_type_vector)
	{
		producer->SetUseMorphemeTypeVector(prop->GetValue().boolVal);
	}
	else if (prop_type == _prop_type::w2v_path)
	{
		producer->SetWordVector(CStringA(prop->GetValue()));
		reload = true;
	}
	else if (prop_type == _prop_type::word_norm)
	{
		producer->SetVectorNormalization(prop->GetValue().boolVal);
	}
	else if (prop_type == _prop_type::src_use)
	{
		if (prop->GetValue().boolVal)
			producer->InsertSourceColumn(prop->index);
		else
			producer->EraseSourceColumn(prop->index);
	}
	else if (prop_type == _prop_type::parse_sentence)
	{
		producer->SetParsingSentence(prop->GetValue().boolVal);
		reload = true;
	}
	else if (prop_type == _prop_type::max_sentence)
	{
		producer->SetMaxSentence(prop->GetValue().intVal);
	}
	else if (prop_type == _prop_type::max_word_per_sentence)
	{
		producer->SetMaxWordPerSentence(prop->GetValue().intVal);
	}
	else if (prop_type == _prop_type::max_word)
	{
		producer->SetMaxWord(prop->GetValue().intVal);
	}
}
