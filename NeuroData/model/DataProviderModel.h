#pragma once

#include "util/UniqueIdFactory.h"

#include "AbstractReaderModel.h"
#include "AbstractProducerModel.h"

#include "lib/NLPUtil/include/WordToVector.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			//enum class _type{input, target};
			typedef std::unordered_map<neuro_u32, AbstractPreprocessorModel*> _model_map;

			enum class _provider_type{learn, predict, both};

			class TextReaderModel;
			class DataProviderModel
			{
			public:
				DataProviderModel(_provider_type provider_type);
				virtual ~DataProviderModel();

				void SetProviderType(_provider_type provider_type) { m_provider_type = provider_type;}
				bool IsPredictProvider() const {
					return m_provider_type == _provider_type::predict || m_provider_type == _provider_type::both;
				}
				bool IsLearnProvider() const {
					return m_provider_type == _provider_type::learn || m_provider_type == _provider_type::both;
				}

				void ClearAll();

				TextReaderModel* ImportCSVFile(const char* filePath);
				AbstractReaderModel* AddReaderModel(_reader_type type);
				bool AddReaderModel(AbstractReaderModel* model);
				const _reader_model_vector& GetReaderVector() const { return m_reader_vector; }

				AbstractProducerModel* AddProducerModel(_producer_type type);
				bool AddProducerModel(AbstractProducerModel* model);
				const _producer_model_vector& GetProducerVector() const { return m_producer_vector; }
				_producer_model_vector& GetProducerVector() { return m_producer_vector; }

				bool DeleteDataModel(AbstractPreprocessorModel* model);
				bool ReplacePreprocessorModel(AbstractPreprocessorModel* old_model, AbstractPreprocessorModel* new_model);

				bool Disconnect(AbstractReaderModel* from, AbstractPreprocessorModel* to);

				AbstractPreprocessorModel* GetDataModel(neuro_32 uid)
				{
					_model_map::const_iterator it = m_model_map.find(uid);
					if (it == m_model_map.end())
						return NULL;
					return it->second;
				}
				const AbstractPreprocessorModel* GetDataModel(neuro_32 uid) const
				{
					return const_cast<DataProviderModel*>(this)->GetDataModel(uid);
				}

			private:
				_provider_type m_provider_type;

				template<class class_type, typename model_type>
				inline class_type* AddDataModel(std::vector<class_type*>& model_vector, model_type type)
				{
					neuro_u32 uid = m_id_factory.CreateId();

					class_type* model = class_type::CreateInstance(*this, type, uid);
					if (model == NULL)
					{
						m_id_factory.RemoveId(uid);
						return NULL;
					}

					model_vector.push_back(model);
					m_model_map[model->uid] = model;
					return model;
				}

				template<class class_type>
				inline bool AddDataModel(std::vector<class_type*>& model_vector, class_type* model)
				{
					if (!m_id_factory.InsertId(model->uid))
						return false;

					model_vector.push_back(model);
					m_model_map[model->uid] = model;
					return true;
				}

				template<class class_type>
				inline bool DelDataModel(std::vector<class_type*>& model_vector, const class_type* model)
				{
					if (model == NULL)
						return false;

					if (!m_id_factory.HasId(model->uid))
						return false;

					m_model_map.erase(model->uid);

					std::vector<class_type*>::iterator it = model_vector.begin();
					for (; it != model_vector.end(); it++)
					{
						if (model == *it)
						{
							model_vector.erase(it);
							m_id_factory.RemoveId(model->uid);
							delete model;
							return true;
						}
					}
					return false;
				}

				template<class class_type>
				inline bool ReplacePreprocessorModel(std::vector<class_type*>& model_vector, class_type* old_model, class_type* new_model)
				{
					if (!m_id_factory.HasId(old_model->uid))
						return false;

					for (neuro_u32 i=0;i<model_vector.size();i++)
					{
						if (old_model == model_vector[i])
						{
							model_vector[i] = new_model;

							if (old_model->uid != new_model->uid)
								m_model_map.erase(old_model->uid);
							m_model_map[new_model->uid] = new_model;

							new_model->SetInput(old_model->GetInput());
							old_model->SetInput(NULL);
							return true;
						}
					}
					return false;
				}
				util::UniqueIdFactory m_id_factory;

				_model_map m_model_map;

				_reader_model_vector m_reader_vector;
				_producer_model_vector m_producer_vector;
			};

			class ProviderModelManager
			{
			public:
				ProviderModelManager();
				virtual ~ProviderModelManager();

				void ClearAll();

				void IntegratedProvider(bool use);

				DataProviderModel& GetPredictProvider() { return m_predict_provider; }
				const DataProviderModel& GetPredictProvider() const { return m_predict_provider; }

				DataProviderModel* GetLearnProvider() { return m_learn_provider; }
				const DataProviderModel* GetLearnProvider() const { return m_learn_provider; }

				const DataProviderModel& GetFinalProvider(bool is_predict) const
				{
					if (!is_predict && m_learn_provider)
						return *m_learn_provider;
					return m_predict_provider;
				}

				AbstractReaderModel* AddReaderModel(AbstractPreprocessorModel& owner, _reader_type type);
				bool DeleteDataModel(AbstractPreprocessorModel* model);

				bool Disconnect(AbstractReaderModel* from, AbstractPreprocessorModel* to);

			private:
				DataProviderModel m_predict_provider;
				DataProviderModel* m_learn_provider;
			};
		}
	}
}
