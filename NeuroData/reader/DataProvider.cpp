#include "stdafx.h"

#include "DataProvider.h"
#include "util/FileUtil.h"

#include "AbstractReader.h"
#include "AbstractProducer.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

DataProvider::DataProvider(InitShareObject& init_object, bool data_noising, bool support_ndf, neuro_u32 batch_size)
	: DataReaderSet(init_object, data_noising, support_ndf, batch_size)
{
}

DataProvider::~DataProvider(void)
{
}

void DataProvider::SetDataSource(const _uid_datanames_map& uid_datanames_map)
{
	this->uid_datanames_map = uid_datanames_map;
}

void DataProvider::SetDataSource(const _uid_mem_data_map& memory_data_map)
{
	this->memory_data_map = memory_data_map;
}

bool DataProvider::Create(const dp::model::DataProviderModel& provider_model)
{
	const dp::model::_producer_model_vector& producer_def_vector = provider_model.GetProducerVector();
	if (producer_def_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"no producer");
		return false;
	}

	const model::_reader_model_vector& reader_model_vector = provider_model.GetReaderVector();
	for (neuro_u32 i = 0; i < reader_model_vector.size(); i++)
	{
		const model::AbstractReaderModel* model = reader_model_vector[i];
		AbstractReader* reader=AbstractReader::CreateInstance(*this, *model);
		if (!reader)
		{
			DEBUG_OUTPUT(L"failed create %uth reader", i);
			return false;
		}
		reader_map[model->uid] = reader;
	}

	const model::_producer_model_vector& producer_model_vector = provider_model.GetProducerVector();
	for (neuro_u32 i = 0; i < producer_model_vector.size(); i++)
	{
		const model::AbstractProducerModel* model = producer_model_vector[i];

		std::string status = util::StringUtil::Format("creating instance of %uth producer", i);
		AbstractProducer* producer = AbstractProducer::CreateInstance(status.c_str(), *this, *model);
		if (!producer)
		{
			DEBUG_OUTPUT(L"failed create %uth producer", i);
			return false;
		}
		producer_map[model->uid] = producer;

		m_producer_vector.push_back(producer);
		m_binding_producer_map[model] = producer;
	}
	return true;
}

bool DataProvider::CreateDirect(const _producer_model_instance_vector& producer_model_instance_vector)
{
	for (neuro_u32 i=0;i<producer_model_instance_vector.size();i++)
	{
		const _PRODUCER_MODEL_INSTANCE& producer = producer_model_instance_vector[i];
		producer_map[producer.model->uid] = producer.instance;
		m_producer_vector.push_back(producer.instance);
		m_binding_producer_map[producer.model] = producer.instance;
	}
	return true;
}

neuro_u64 DataProvider::GetDataCount() const
{
	if (m_producer_vector.size() == 0)
		return 0;

	neuro_u64 data_count = m_producer_vector[0]->GetDataCount();
	for (neuro_u32 i = 1; i<m_producer_vector.size(); i++)
		data_count = min(data_count, m_producer_vector[i]->GetDataCount());

	return data_count;
}

AbstractProducer* DataProvider::FindBindingProducer(const NetworkBindingModel* binding)
{
	_binding_producer_map::const_iterator it = m_binding_producer_map.find(binding);
	if(it == m_binding_producer_map.end())
		return NULL;

	return it->second;
}

bool DataProvider::Preload()
{
	for (neuro_u32 i = 0; i<m_producer_vector.size(); i++)
	{
		AbstractProducer* producer = m_producer_vector[i];
		if(!producer->Preload())
		{
			DEBUG_OUTPUT(L"failed %u th", i);
			return false;
		}
	}

	return true;
}
