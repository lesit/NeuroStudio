#pragma once

#include "AbstractLayer.h"
#include <unordered_map>

#include "../nsas/NeuralNetworkEntrySpec.h"

#include "NeuroData/model/AbstractProducerModel.h"

#include "HiddenLayer.h"
namespace np
{
	namespace network
	{
		class InputLayer : public AbstractLayer
		{
		public:
			InputLayer(neuro_u32 uid, const tensor::TensorShape& ts)
				: AbstractLayer(uid)
			{
				SetValidDataShape(ts);
			}

			virtual ~InputLayer() {}

			_binding_model_type GetBindingModelType() const override { return _binding_model_type::network_input_layer; }

			network::_layer_type GetLayerType() const override {
				return network::_layer_type::input;
			}

			void ChangedBindingDataShape() override
			{
				const _neuro_binding_model_set& binding_set = GetBindingSet();
				for (_neuro_binding_model_set::const_iterator it = binding_set.begin(); it != binding_set.end(); it++)\
				{
					const NetworkBindingModel* binding = *it;
					dp::model::AbstractProducerModel* producer = (dp::model::AbstractProducerModel*)binding;
					if (producer->GetBindingModelType() == _binding_model_type::data_producer
						&& producer->IsInPredictProvider())	// predict provider에 등록된 것을 사용
					{
						SetDataShape(producer->GetDataShape());
						break;
					}
				}
			}

			void SetDataShape(const tensor::DataShape& ts)
			{
				SetValidDataShape(ts);
				CheckOutputTensor();
			}
		protected:
			inline void SetValidDataShape(const tensor::DataShape& ts)
			{
				m_out_ts = ts;
				if (m_out_ts.size()<3)
					m_out_ts.resize(3, neuro_u32(1));
			}

			tensor::TensorShape MakeOutTensorShape() const { return m_out_ts; }
		};
		typedef std::vector<InputLayer*> _input_layer_vector;
	}
}
