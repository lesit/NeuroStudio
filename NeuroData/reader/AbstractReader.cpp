#include "stdafx.h"

#include "AbstractReader.h"

#include "BinaryReader.h"
#include "TextReader.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

AbstractReader* AbstractReader::CreateInstance(DataReaderSet& reader_set, const model::AbstractReaderModel& model)
{
	AbstractReader* reader = NULL;
	if (model.GetReaderType() == model::_reader_type::binary)
		reader = new BinaryReader(model);
	else if (model.GetReaderType() == model::_reader_type::text)
		reader = new TextReader(model);

	if (reader == NULL)
		return NULL;

	if (!reader->Create(reader_set))
	{
		delete reader;
		return NULL;
	}
	return reader;
}
