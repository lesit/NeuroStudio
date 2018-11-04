#pragma once

#include "DataReaderSet.h"

#include "../model/DataProviderModel.h"

#include "AbstractProducer.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			typedef std::vector<AbstractProducer*> _data_producer_vector;

			typedef std::unordered_map<const NetworkBindingModel*, AbstractProducer*> _binding_producer_map;

			struct _PRODUCER_MODEL_INSTANCE
			{
				const model::AbstractProducerModel* model;
				AbstractProducer* instance;
			};
			typedef std::vector<_PRODUCER_MODEL_INSTANCE> _producer_model_instance_vector;

			class DataProvider : public DataReaderSet
			{
			public:
				DataProvider(InitShareObject& init_object, bool data_noising, bool support_ndf, neuro_u32 batch_size);
				virtual ~DataProvider();

				void SetDataSource(const _uid_datanames_map& uid_datanames_map);
				void SetDataSource(const _uid_mem_data_map& memory_data_map);
				bool Create(const dp::model::DataProviderModel& provider_model);

				bool CreateDirect(const _producer_model_instance_vector& producer_model_instance_vector);

				AbstractProducer* FindBindingProducer(const NetworkBindingModel* binding);

				neuro_u64 GetDataCount() const;

				bool Preload();

			protected:
				_data_producer_vector m_producer_vector;
				_binding_producer_map m_binding_producer_map;
			};
		}
	}
}
