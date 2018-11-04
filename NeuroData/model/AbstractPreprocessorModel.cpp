#include "AbstractPreprocessorModel.h"

#include "DataProviderModel.h"

#include "AbstractReaderModel.h"

using namespace np;
using namespace np::dp;
using namespace np::dp::model;

AbstractPreprocessorModel::AbstractPreprocessorModel(DataProviderModel& provider, neuro_u32 _uid)
	: m_provider(provider), uid(_uid)
{
	m_input = NULL;
}

AbstractPreprocessorModel::~AbstractPreprocessorModel()
{
	_preprocessor_model_set::iterator it = m_output_set.begin();
	for (; it != m_output_set.end(); it++)
		(*it)->m_input = NULL;
}

bool AbstractPreprocessorModel::IsInPredictProvider() const
{
	return m_provider.IsPredictProvider();
}

_input_source_type AbstractPreprocessorModel::GetInputSourceType() const 
{
	if (m_input)
		return m_input->GetInputSourceType();

	return _input_source_type::none;
}

bool AbstractPreprocessorModel::AvailableAttachDeviceInput() const {
	return GetInputSourceType() != _input_source_type::none;
}

bool AbstractPreprocessorModel::AvailableInput(_reader_type reader_type) const
{
	std::unordered_set<_reader_type> types = GetAvailableInputReaderTypeSet();
	return types.find(reader_type) != types.end();
}

void AbstractPreprocessorModel::SetInput(AbstractReaderModel* input)
{
	if (input == m_input)
		return;

	if (m_input != NULL)
		m_input->m_output_set.erase(this);

	if (input != NULL)
	{
		if (AvailableInput(input->GetReaderType()))
			input->m_output_set.insert(this);
		else
			input = NULL;
	}

	m_input = input; 

	ChangedProperty();
}
