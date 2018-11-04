#include "TextReaderModel.h"

#include "util/np_util.h"
#include "util/randoms.h"

using namespace np::dp;
using namespace np::dp::model;

TextReaderModel::TextReaderModel(DataProviderModel& provider, neuro_u32 uid)
: AbstractReaderModel(provider, uid)
{
	m_skip_count = 0;
	m_is_reverse = false;

	m_column_delimiter = new ExtTextColumnTokenDelimiter;
}

TextReaderModel::~TextReaderModel()
{
	delete m_column_delimiter;
}

void TextReaderModel::ChangeDelimiterType(_delimiter_type type)
{
	if (m_column_delimiter->GetType() == type)
		return;

	delete m_column_delimiter;
	switch (type)
	{
	case _delimiter_type::length:
		m_column_delimiter = new TextColumnLengthDelimiter;
		break;
	default:
		m_column_delimiter = new ExtTextColumnTokenDelimiter;
	}
}

void TextReaderModel::SetColumnCount(neuro_u32 count)
{
	if (m_column_delimiter->GetType() == dp::_delimiter_type::length)
		((TextColumnLengthDelimiter*)m_column_delimiter)->lengh_vector.resize(count, 0);
	else
		((ExtTextColumnTokenDelimiter*)m_column_delimiter)->m_column_count = count;

	ChangedProperty();
}

neuro_u32 TextReaderModel::GetColumnCount() const
{
	if (m_column_delimiter->GetType() == dp::_delimiter_type::length)
		return ((TextColumnLengthDelimiter*)m_column_delimiter)->lengh_vector.size();
	else
		return ((ExtTextColumnTokenDelimiter*)m_column_delimiter)->m_column_count;
}

bool TextReaderModel::ImportCSV(device::FileDeviceAdaptor& input_device)
{
	TextColumnTokenDelimiter delimiter;
	return Import(input_device, delimiter);
}

bool TextReaderModel::Import(device::FileDeviceAdaptor& input_device, const TextColumnTokenDelimiter& delimiter)
{
	dp::TextParsingReader* parser = dp::TextParsingReader::CreateInstance(delimiter);
	if (!parser->SetInputDevice(input_device))
	{
		delete parser;
		return false;
	}

	if (m_column_delimiter->GetType() != delimiter.GetType())
	{
		delete m_column_delimiter;
		m_column_delimiter = new ExtTextColumnTokenDelimiter;
	}
	*m_column_delimiter = delimiter;

	std_string_vector column_vector;
	bool eof;
	parser->ReadContent(&column_vector, eof);

	((ExtTextColumnTokenDelimiter*)m_column_delimiter)->m_column_count = column_vector.size();

	m_is_reverse = false;
	m_skip_count=1;	// header ¶§¹®¿¡
	m_imported_source = input_device.GetDeviceName();

	delete parser;
	return true;
}
