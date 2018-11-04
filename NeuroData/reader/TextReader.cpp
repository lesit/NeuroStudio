#include "stdafx.h"

#include "TextReader.h"

#include "util/np_util.h"
#include "util/randoms.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

TextReader::TextReader(const dp::model::AbstractReaderModel& model)
	: m_model((const dp::model::TextReaderModel&)model)
{
	_u32_set index_set;
	m_model.GetUsingSourceIndexSet(index_set);

	for(_u32_set::const_iterator it=index_set.begin();it!=index_set.end();it++)
	{
		if ((*it) >= m_model.GetColumnCount())
			break;
		m_using_index_vector.push_back(*it);
	}

	m_position = 0;
}

TextReader::~TextReader()
{
}

bool TextReader::Create(DataReaderSet& reader_set)
{
	if (!AttachInputDevices(m_model, reader_set))
		return false;

	for (neuro_u32 i = 0; i < m_device_vector.size(); i++)
	{
		device::DeviceAdaptor* device = m_device_vector[i];

		JobSignalSender job(reader_set.init_object.GetLongTimeJobSignal(), 0, std::string("creating reader instance : ").append(device->GetDeviceName()));

		dp::TextParsingReader* reader = dp::TextParsingReader::CreateInstance(m_model.GetDelimiter(), &m_using_index_vector);
		if (!reader)
			return false;
		if (!reader->SetInputDevice(*device))
		{
			job.failure();
			delete reader;

			continue;
		}

		neuro_size_t count = reader->ReadAllContents(m_model.GetSkipFirstCount(), m_model.IsReverse(), m_content_vector);
		if (count == 0)
		{
			job.failure();
			DEBUG_OUTPUT(L"no data.");
			delete reader;

			continue;
		}
	}
	return true;
}

bool TextReader::Read(neuro_size_t pos)
{
	if (m_position >= m_content_vector.size())
		return false;

	m_position = pos;
	return true;
}
