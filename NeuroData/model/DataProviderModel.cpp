#include "DataProviderModel.h"

#include "NlpProducerModel.h"

#include "util/FileUtil.h"

using namespace np::dp;
using namespace np::dp::model;

DataProviderModel::DataProviderModel(_provider_type provider_type)
{
	m_provider_type = provider_type;
}

DataProviderModel::~DataProviderModel()
{
	ClearAll();
}

void DataProviderModel::ClearAll()
{
	m_id_factory.RemoveAll();
	m_model_map.clear();

	for (neuro_u32 i = 0; i < m_reader_vector.size(); i++)
		delete m_reader_vector[i];
	m_reader_vector.clear();

	for (neuro_u32 i = 0; i < m_producer_vector.size(); i++)
		delete m_producer_vector[i];
	m_producer_vector.clear();
}

TextReaderModel* DataProviderModel::ImportCSVFile(const char* filePath)
{
	device::FileDeviceAdaptor device;
	if (!device.Create(filePath, true, false, true))
		return NULL;

	TextReaderModel* model = (TextReaderModel*) AddReaderModel(_reader_type::text);
	if (!model)
		return NULL;

	if (!model->ImportCSV(device))
	{
		delete model;
		return NULL;
	}

	AddReaderModel(model);
	return model;
}

AbstractReaderModel* DataProviderModel::AddReaderModel(_reader_type type)
{
	return AddDataModel<AbstractReaderModel, _reader_type>(m_reader_vector, type);
}

bool DataProviderModel::AddReaderModel(AbstractReaderModel* model)
{
	return AddDataModel<AbstractReaderModel>(m_reader_vector, model);
}

AbstractProducerModel* DataProviderModel::AddProducerModel(_producer_type type)
{
	return AddDataModel<AbstractProducerModel, _producer_type>(m_producer_vector, type);
}

bool DataProviderModel::AddProducerModel(AbstractProducerModel* model)
{
	return AddDataModel<AbstractProducerModel>(m_producer_vector, model);
}

bool DataProviderModel::DeleteDataModel(AbstractPreprocessorModel* model)
{
	if (!model)
		return false;

	if (model->GetInput())
		Disconnect(model->GetInput(), model);

	if(model->GetModelType()==_model_type::reader)
		return DelDataModel<AbstractReaderModel>(m_reader_vector, (const AbstractReaderModel*) model);
	else if(model->GetModelType() == _model_type::producer)
		return DelDataModel<AbstractProducerModel>(m_producer_vector, (const AbstractProducerModel*)model);
	return false;
}

bool DataProviderModel::ReplacePreprocessorModel(AbstractPreprocessorModel* old_model, AbstractPreprocessorModel* new_model)
{
	if (old_model == NULL || new_model == NULL || old_model->GetModelType() != new_model->GetModelType())
		return false;

	if (old_model->GetModelType() == _model_type::reader)
	{
		if (!((AbstractReaderModel*)old_model)->AvailableChangeType(((AbstractReaderModel*)new_model)->GetReaderType()))
			return false;

		if (!ReplacePreprocessorModel<AbstractReaderModel>(m_reader_vector, (AbstractReaderModel*)old_model, (AbstractReaderModel*)new_model))
			return false;

		_preprocessor_model_set output_set = old_model->GetOutputSet();	// SetInput을 통해서 old_model의 m_output_set이 바뀌므로 &참조가 아닌 복사를 해야 한다.
		_preprocessor_model_set::const_iterator it_out = output_set.begin();
		for (; it_out != output_set.end(); it_out++)
			(*it_out)->SetInput((AbstractReaderModel*)new_model);
	}
	else if (old_model->GetModelType() == _model_type::producer)
	{
		if (!((AbstractProducerModel*)old_model)->AvailableChangeType(((AbstractProducerModel*)new_model)->GetProducerType()))
			return false;

		if (!ReplacePreprocessorModel<AbstractProducerModel>(m_producer_vector, (AbstractProducerModel*)old_model, (AbstractProducerModel*)new_model))
			return false;

		((AbstractProducerModel*)new_model)->RemoveAllBinding();

		const _neuro_binding_model_set& binding_set = ((AbstractProducerModel*)old_model)->GetBindingSet();
		_neuro_binding_model_set::const_iterator it = binding_set.begin();
		for (; it != binding_set.end(); it++)
		{
			((AbstractProducerModel*)new_model)->AddBinding(*it);
			(*it)->ChangedBindingDataShape();	// network layer한테 알려야 한다.
		}

		((AbstractProducerModel*)old_model)->RemoveAllBinding();
	}
	else
		return false;

	return true;
}

bool DataProviderModel::Disconnect(AbstractReaderModel* from, AbstractPreprocessorModel* to)
{
	if (from == NULL || to == NULL)
		return false;

	to->SetInput(from);

	// 입력 reader가 출력을 가지고 있지 않으면 삭제 해야함
	while(from && from->GetOutputSet().size()==0)
	{
		to = from;
		from = from->GetInput();
		DelDataModel<AbstractReaderModel>(m_reader_vector, (const AbstractReaderModel*)to);
	}
	return true;
}

ProviderModelManager::ProviderModelManager()
	: m_predict_provider(_provider_type::predict)
{
	m_learn_provider = NULL;
	IntegratedProvider(false);	// 일반적으로 독립적이다.
}

ProviderModelManager::~ProviderModelManager()
{
	delete m_learn_provider;
}

void ProviderModelManager::ClearAll()
{
	if (m_learn_provider)
		m_learn_provider->ClearAll();
	m_predict_provider.ClearAll();
}

void ProviderModelManager::IntegratedProvider(bool is_integrated)
{
	if (is_integrated)
	{
		delete m_learn_provider;
		m_learn_provider = NULL;

		m_predict_provider.SetProviderType(_provider_type::both);
	}
	else
	{
		if (m_learn_provider == NULL)
			m_learn_provider = new DataProviderModel(_provider_type::learn);

		m_predict_provider.SetProviderType(_provider_type::predict);
	}
}

AbstractReaderModel* ProviderModelManager::AddReaderModel(AbstractPreprocessorModel& owner, _reader_type type)
{
	AbstractReaderModel* reader = owner.GetProvider().AddReaderModel(type);
	if (reader != NULL)
		owner.SetInput(reader);

	return reader;
}

bool ProviderModelManager::DeleteDataModel(AbstractPreprocessorModel* model)
{
	return model->GetProvider().DeleteDataModel(model);
}

bool ProviderModelManager::Disconnect(AbstractReaderModel* from, AbstractPreprocessorModel* to)
{
	return to->GetProvider().Disconnect(from, to);
}
