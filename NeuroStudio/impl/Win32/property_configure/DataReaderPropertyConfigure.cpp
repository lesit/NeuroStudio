#include "stdafx.h"

#include "DataReaderPropertyConfigure.h"

using namespace property;

std::wstring DataReaderPropertyConfigure::GetPropertyName() const
{
	return L"Data Reader";
}

const wchar_t* DataReaderPropertyConfigure::GetSubTypeString(neuro_u32 type) const
{
	return ToReaderString((_reader_type)type);
}

_reader_type DataReaderPropertyConfigure::GetModelSubType(const AbstractReaderModel* model) const
{
	return model->GetReaderType();
}

void DataReaderPropertyConfigure::CompositeProperties()
{
	AbstractReaderModel* model = GetModel();

	CModelGridProperty* column_count_property = new CModelGridProperty(L"Column count", (_variant_t)model->GetColumnCount()
		, NULL, (DWORD_PTR)_prop_type::column_count);
	column_count_property->EnableSpinControl(TRUE, 0, neuro_last16);

	if (model->GetReaderType() == _reader_type::text)
	{
		TextReaderModel* text_reader = (TextReaderModel*)model;

		CModelGridProperty* prop = new CModelGridProperty(L"Skip count"
			, (_variant_t)text_reader->GetSkipFirstCount()
			, L"skip first some lines when reading"
			, (DWORD_PTR)_prop_type::skip);
		m_list_ctrl.AddProperty(prop);

		prop = new CModelGridProperty(L"Reverse", (_variant_t)text_reader->IsReverse()
			, L"reverse reading", (DWORD_PTR)_prop_type::reverse);
		m_list_ctrl.AddProperty(prop);

		dp::TextColumnDelimiter& delimiter = text_reader->GetDelimiter();

		CModelGridProperty* delimiter_type_prop = new CModelGridProperty(L"Delimiter Type", (_variant_t)_delimiter_type_string[(int)delimiter.GetType()]
			, L"delimiter to parse columns", (DWORD_PTR)_prop_type::delimiter_type);
		m_list_ctrl.AddProperty(delimiter_type_prop);

		for (neuro_u32 type = 0; type < _countof(_delimiter_type_string); type++)
			delimiter_type_prop->AddOption(_delimiter_type_string[type]);
		delimiter_type_prop->AllowEdit(FALSE);

		if (text_reader->GetDelimiter().GetType() == _delimiter_type::token)
		{
			CModelGridProperty* delimiter_prop = new CModelGridProperty(L"Delimiter", (DWORD_PTR)_prop_type::delimiters);
			m_list_ctrl.AddProperty(delimiter_prop);

			ExtTextColumnTokenDelimiter& text_delimiter = (ExtTextColumnTokenDelimiter&)text_reader->GetDelimiter();

			std::wstring token = util::StringUtil::MultiByteToWide(util::StringUtil::TransformAsciiToText(text_delimiter.content_token));
			CModelGridProperty* child_prop = new CModelGridProperty(L"Content Token", (_variant_t)token.c_str()
				, L"token to parse contents", (DWORD_PTR)_prop_type::content_token);
			delimiter_prop->AddSubItem(child_prop);

			child_prop = new CModelGridProperty(L"Double Quotes", (_variant_t)text_delimiter.double_quote
				, L"has duoble quotes like csv format", (DWORD_PTR)_prop_type::double_quote);
			delimiter_prop->AddSubItem(child_prop);

			child_prop = new CModelGridProperty(L"Column Token", (_variant_t)util::StringUtil::MultiByteToWide(TokenVectorToString(text_delimiter.column_token_vector)).c_str()
				, L"token to parse columns", (DWORD_PTR)_prop_type::column_token);
			delimiter_prop->AddSubItem(child_prop);
		}

		m_list_ctrl.AddProperty(column_count_property);

		if (text_reader->GetDelimiter().GetType() == _delimiter_type::length)
		{
			CMFCPropertyGridProperty* column_list_prop = new CMFCPropertyGridProperty(L"Column list");
			m_list_ctrl.AddProperty(column_list_prop);

			TextColumnLengthDelimiter& fixlen_delimiter = (TextColumnLengthDelimiter&)text_reader->GetDelimiter();

			for (neuro_u32 i = 0; i < fixlen_delimiter.lengh_vector.size(); i++)
			{
				CModelGridProperty* column_prop = new CModelGridProperty(util::StringUtil::Format<wchar_t>(L"%u length", i).c_str()
					, (_variant_t)fixlen_delimiter.lengh_vector[i]
					, L" Column length"
					, (DWORD_PTR)_prop_type::fixed_len);

				column_prop->index = i;

				column_list_prop->AddSubItem(column_prop);
			}
		}
	}
	else if (model->GetReaderType() == _reader_type::binary)
	{
		m_list_ctrl.AddProperty(column_count_property);

		CMFCPropertyGridProperty* column_list_prop = new CMFCPropertyGridProperty(L"Column list");
		m_list_ctrl.AddProperty(column_list_prop);

		BinaryReaderModel* reader = (BinaryReaderModel*)model;
		const _data_type_vector& type_vector = reader->GetTypeVector();
		for (neuro_u32 i = 0, n = type_vector.size(); i < n; i++)
		{
			std::wstring column_label = util::StringUtil::Format<wchar_t>(L"%u Column data type", i);

			CModelGridProperty* column_prop = new CModelGridProperty(column_label.c_str()
				, (_variant_t)_data_type_string[(int)type_vector[i]]
				, L"", (DWORD_PTR)_prop_type::data_type);

			column_list_prop->AddSubItem(column_prop);

			column_prop->index = i;

			for (neuro_u32 type = 0; type < _countof(_data_type_string); type++)
				column_prop->AddOption(_data_type_string[type]);
			column_prop->AllowEdit(FALSE);
		}
	}

	m_list_ctrl.ExpandAll();
}

void DataReaderPropertyConfigure::PropertyChanged(CModelGridProperty* prop, bool& reload) const
{
	_prop_type text_prop_type = (_prop_type) prop->GetData();

	AbstractReaderModel* model = GetModel();

	if (text_prop_type == _prop_type::column_count)
	{
		model->SetColumnCount(prop->GetValue().uintVal);

		reload = true;
		return;
	}

	if (model->GetReaderType() == _reader_type::text)
	{
		TextReaderModel* text_reader = (TextReaderModel*)model;
		if (text_prop_type == _prop_type::skip)
		{
			text_reader->SetSkipFirstCount(prop->GetValue().uintVal);
		}
		else if (text_prop_type == _prop_type::reverse)
		{
			text_reader->SetReverse(prop->GetValue().boolVal);
		}
		else if (text_prop_type == _prop_type::delimiter_type)
		{
			CString strTypeString(prop->GetValue());
			text_reader->ChangeDelimiterType(ToDelimiterType(strTypeString));

			reload = true;
		}
		else
		{
			if (((TextReaderModel*)model)->GetDelimiter().GetType() == _delimiter_type::token)
			{
				TextColumnTokenDelimiter& fixlen_delimiter = (TextColumnTokenDelimiter&)((TextReaderModel*)model)->GetDelimiter();
				if (text_prop_type == _prop_type::content_token)
				{
					CString strValue(prop->GetValue());
					fixlen_delimiter.content_token = util::StringUtil::TransformTextToAscii(util::StringUtil::WideToMultiByte((const wchar_t*)strValue));
				}
				else if (text_prop_type == _prop_type::double_quote)
				{
					fixlen_delimiter.double_quote =prop->GetValue().boolVal;
				}
				else if (text_prop_type == _prop_type::column_token)
				{
					CString strValue(prop->GetValue());
					StringToTokenVector(util::StringUtil::WideToMultiByte((const wchar_t*)strValue), fixlen_delimiter.column_token_vector);
				}
			}
			else if (((TextReaderModel*)model)->GetDelimiter().GetType() == _delimiter_type::length)
			{
				if (text_prop_type == _prop_type::fixed_len)
				{
					TextColumnLengthDelimiter& fixlen_delimiter = (TextColumnLengthDelimiter&)text_reader->GetDelimiter();
					fixlen_delimiter.lengh_vector[prop->index] = prop->GetValue().uintVal;
				}
			}
		}
	}
	else if (model->GetReaderType() == _reader_type::binary)
	{
		if (text_prop_type == _prop_type::data_type)
		{
			BinaryReaderModel* reader = (BinaryReaderModel*)model;

			_data_type_vector& type_vector = reader->GetTypeVector();
			type_vector[prop->index] = ToDataType(CString(prop->GetValue()));
		}
	}
}
