#include "AbstractProducerModel.h"

#include "ImageFileProducerModel.h"
#include "IncreasePredictProducerModel.h"
#include "MnistProducerModel.h"
#include "NlpProducerModel.h"
#include "NumericProducerModel.h"
#include "WindllProducerModel.h"

using namespace np::dp::model;

AbstractProducerModel* AbstractProducerModel::CreateInstance(DataProviderModel& provider, _producer_type type, neuro_u32 uid)
{
	AbstractProducerModel* model = NULL;
	switch (type)
	{
	case _producer_type::image_file:
		model = new ImageFileProducerModel(provider, uid);
		break;
	case _producer_type::mnist_img:
		model = new MnistImageProducerModel(provider, uid);
		break;
	case _producer_type::mnist_label:
		model = new MnistLabelProducerModel(provider, uid);
		break;
	case _producer_type::nlp:
		model = new NlpProducerModel(provider, uid);
		break;
	case _producer_type::numeric:
		model = new NumericProducerModel(provider, uid);
		break;
	case _producer_type::increase_predict:
		model = new IncreasePredictProducerModel(provider, uid);
		break;
	}
	return model;
}

std::unordered_set<_reader_type> AbstractProducerModel::GetAvailableInputReaderTypeSet(_producer_type owner_type)
{
	switch (owner_type)
	{
	case _producer_type::numeric:
	case _producer_type::increase_predict:
		return{ _reader_type::binary, _reader_type::text/*, _reader_type::database*/ };
	case _producer_type::nlp:
		return{ _reader_type::text/*, _reader_type::database*/ };
	}
	return{/*_reader_type::database*/ };
}

AbstractProducerModel::AbstractProducerModel(DataProviderModel& provider, neuro_u32 uid)
	: AbstractPreprocessorModel(provider, uid)
{
	m_start = 0;
	m_scale_min = -1.f;
	m_scale_max = 1.f;
}

void AbstractProducerModel::GetAvailableChangeTypes(std::vector<_producer_type>& type_vector) const
{
	for (neuro_u32 i = 0; i < _countof(_producer_type_string); i++)
	{
		if (AvailableChangeType((_producer_type)i))
			type_vector.push_back((_producer_type)i);
	}
}

neuro_u32 AbstractProducerModel::GetLabelOutCount() const 
{
	if (GetLabelOutType() != _label_out_type::label_dir) return 0;

	return m_label_dir_vector.size() >= 2 ? m_label_dir_vector.size() : 0;
}

void AbstractProducerModel::SetLabelOutCount(neuro_u32 scope)
{
	if (GetLabelOutType() != _label_out_type::label_dir)
		return;

	if (scope < 2) scope = 0;

	neuro_u32 old = m_label_dir_vector.size();
	m_label_dir_vector.resize(scope);
	for (neuro_u32 i = old; i < m_label_dir_vector.size(); i++)
		m_label_dir_vector[i] = util::StringUtil::Format<char>("data_%u", i);
}
