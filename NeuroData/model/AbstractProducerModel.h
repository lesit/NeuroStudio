#pragma once

#include "AbstractPreprocessorModel.h"
#include "AbstractReaderModel.h"
#include "NetworkBindingModel.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			enum class _producer_type {numeric, nlp, increase_predict, image_file, mnist_img, mnist_label, imagenet, cifar, java, windll, unknown };
			static const wchar_t* _producer_type_string[] = { L"Numeric", L"Nlp", L"Increase predict", L"Image file", L"Mnist image", L"Mnist label", /*L"Imagenet", L"Cifar", L"Java", L"Windows dll"*/ };
			static _producer_type ToProducerType(const wchar_t* name)
			{
				for (neuro_u32 type = 0; type < _countof(_producer_type_string); type++)
					if (wcscmp(name, _producer_type_string[type]) == 0)
						return (_producer_type)type;

				return _producer_type::unknown;
			}
			static const wchar_t* ToProducerString(_producer_type type)
			{
				if ((neuro_u32)type >= _countof(_producer_type_string))
					return L"";

				return _producer_type_string[(neuro_u32)type];
			}

			enum class _ndf_dim_type { all_fix, variable_except_last, all_variable };

			enum class _label_out_type { none, label_dir, direct_def };

			// 나중에 실제로 Data Stream 처리기가 분리될때 각 함수를 export 시킬수 있어야 한다.
			// 즉, Neuro Studio에서 이 kernel을 호출시키는 관점으로 봐야 한다.
			// 따라서, kernel에 이 모든것을 다 담던지, 아니면 stream 처리기를 다른 모듈로 분리해서
			class AbstractProducerModel : public AbstractPreprocessorModel, public NetworkBindingModel
			{
			protected:
				AbstractProducerModel(DataProviderModel& provider, neuro_u32 uid);

			public:
				virtual ~AbstractProducerModel() {}

				_binding_model_type GetBindingModelType() const override { return _binding_model_type::data_producer; }

				neuro_u32 GetUniqueID() const override { return uid; }

				static AbstractProducerModel* CreateInstance(DataProviderModel& provider, _producer_type type, neuro_u32 uid);
				static std::unordered_set<_reader_type> GetAvailableInputReaderTypeSet(_producer_type owner_type);

				_model_type GetModelType() const override { return _model_type::producer; }

				std::unordered_set<_reader_type> GetAvailableInputReaderTypeSet() const override
				{
					return GetAvailableInputReaderTypeSet(GetProducerType());
				}

				inline bool AvailableChangeType(_producer_type type) const
				{
					if (m_input)	// 입력이 있을 경우 새로운 type이 가능한 입력 type인지 검사
					{
						std::unordered_set<_reader_type> available_set = GetAvailableInputReaderTypeSet(type);
						if (available_set.find(m_input->GetReaderType()) == available_set.end())
							return false;
					}
					return true;
				}
				void GetAvailableChangeTypes(std::vector<_producer_type>& type_vector) const;

				virtual _producer_type GetProducerType() const = 0;
				virtual bool AvailableToInputLayer() const { return true; }
				virtual bool AvailableToOutputLayer() const { return true; }

				virtual bool IsImageProcessingProducer() const { return false; }

				virtual bool IsDynamicProducer() const { return false; }

				virtual bool AvailableUsingPredict() const { return true; }

				virtual tensor::DataShape GetDataShape() const = 0;

				virtual bool SupportNdfClone() const { return false; }
				virtual std::string MakeNdfPath(const std::string& source_name) const { return ""; }
				virtual _ndf_dim_type GetNdfDimType() const { return _ndf_dim_type::all_fix; }

				void SetScale(neuron_value min, neuron_value max) { m_scale_min = min; m_scale_max = max; }
				neuron_value GetMinScale() const { return m_scale_min; }
				neuron_value GetMaxScale() const { return m_scale_max; }

				void SetStartPosition(neuro_u32 pos) { m_start = pos; }
				neuro_u32 GetStartPosition() const { return max(m_start, GetAvailableStartPosition()); }

				virtual void ChangedProperty() override
				{
					ChangedDataShape();
				}

				virtual neuro_u32 GetAvailableStartPosition() const { return 0; }

				// label을 생성할수 있는지. 1이상이면 생성할수 있고 범위가 된다.
				virtual _label_out_type GetLabelOutType() const { return _label_out_type::none; }

				virtual neuro_u32 GetLabelOutCount() const;

				virtual void SetLabelOutCount(neuro_u32 scope);
				const std_string_vector& GetLabelDirVector() const { return m_label_dir_vector; }
				void SetLabelDirVector(const std_string_vector& def_vector) { m_label_dir_vector = def_vector; }

			private:
				neuron_value m_scale_min;
				neuron_value m_scale_max;

				neuro_u32 m_start;

				std_string_vector m_label_dir_vector;
			};

			typedef std::vector<AbstractProducerModel*> _producer_model_vector;

			class ImageProcessingProducerModel : public AbstractProducerModel
			{
			public:
				ImageProcessingProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: AbstractProducerModel(provider, uid)
				{}
				bool IsImageProcessingProducer() const { return true; }
			};

			class DynamicProducerModel : public AbstractProducerModel
			{
			public:
				DynamicProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: AbstractProducerModel(provider, uid)
				{}

				virtual bool IsDynamicProducer() const { return true; }
			};
		}
	}
}
