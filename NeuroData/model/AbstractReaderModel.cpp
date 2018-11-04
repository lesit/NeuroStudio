#include "AbstractReaderModel.h"

#include "TextReaderModel.h"
#include "BinaryReaderModel.h"

using namespace np::dp;
using namespace np::dp::model;

AbstractReaderModel* AbstractReaderModel::CreateInstance(DataProviderModel& provider, _reader_type type, neuro_u32 uid)
{
	AbstractReaderModel* model = NULL;
	switch (type)
	{
	case _reader_type::text:
		model = new TextReaderModel(provider, uid);
		break;
	case _reader_type::binary:
		model = new BinaryReaderModel(provider, uid);
		break;
	}
	return model;
}

void AbstractReaderModel::GetUsingSourceIndexSet(_u32_set& index_set) const
{
	_preprocessor_model_set::const_iterator it = m_output_set.begin();
	for (; it != m_output_set.end(); it++)
	{
		AbstractPreprocessorModel* output = *it;
		output->GetUsingSourceIndexSet(index_set);
	}
}
