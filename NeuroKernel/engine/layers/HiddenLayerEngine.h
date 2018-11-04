#pragma once

#include "AbstractLayerEngine.h"

#include "tensor/tensor_shape.h"

#include "../LayerData.h"

#include "../backend/activations.h"

#include "../../nsas/NeuralNetworkEntrySpec.h"
#include "../../network/HiddenLayer.h"

namespace np
{
	namespace engine
	{
		// 보통은 layer 로 하는 것을 여기에서는 더 작은 그룹인 layer 으로 연결하도록 해서
		// 하나의 layer에 한개 이상의 layer을 가지고, 두 layer 간의 연결은 layer 간의 부분 연결을 할 수 있도록 한다.
		namespace layers
		{
			struct _INPUT_INFO
			{
				AbstractLayerEngine* engine;
				nsas::_SLICE_INFO slice_info;
			};
			typedef std::vector<_INPUT_INFO> _input_vector;

			// 현재는 HiddenLayerEngine 만 상속 받고 있으나, 나중엔 sample normalize layer 등도 생긴다.
			class HiddenLayerEngine : public AbstractLayerEngine
			{
			public:
				virtual ~HiddenLayerEngine();

				bool Initialize(const _input_vector& input_vector);

				bool Propagation(bool bTrain, neuro_u32 batch_size);
				bool Backpropagation(neuro_u32 batch_size);

				network::_layer_type GetLayerType() const override { return m_layer.GetLayerType(); }
				virtual const wchar_t* GetLayerName() const = 0;

				virtual neuro_size_t GetOnehotSizePerBatch() const { return 0; }

				activation::ActivationFunction* GetActivation() { return m_activation; }

				std::pair<neuron_value, neuron_value> GetOutputScale() const override;

				const tensor::TensorShape& GetInputDataShape() const{ return m_in_ts; }

				const _input_vector& GetInputLayers() const{ return m_input_vector; }

				const _NEURO_TENSOR_DATA& GetErrorBuffer() const;

				inline const _layer_data_vector& GetInnerDataVector() const { return m_inner_data_vector; }
				inline _layer_data_vector& GetInnerDataVector() { return m_inner_data_vector; }

				const nsas::_LAYER_STRUCTURE_UNION& GetEntry() const{ return m_entry; }

				virtual neuro_u32 Get1MultiplierSize() const{ return 0; }
				virtual neuro_u32 Get1MultiplierSizePerBatch() const{ return 0; }

				network::_weight_init_type GetWeightInitType(network::_layer_data_type wtype) const
				{
					return ((HiddenLayer&)m_layer).GetWeightInitType(wtype);
				}
			protected:
				virtual bool OnInitialized() { return true; }

				virtual bool MustHaveInput() const { return true; }
				virtual bool HasOneInput() const { return true; }

				bool GetInputData(const _INPUT_INFO& input, _NEURO_TENSOR_DATA& buffer) const;

				virtual bool Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output);
				virtual bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) {
					return false;
				}

				virtual bool Backward(neuro_u32 batch_size);

				virtual bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error){
					return true;
				}

				virtual bool BackwardWeight(neuro_u32 index
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& input_data
					, const _VALUE_VECTOR& grad_weight){
					return true;
				}

				virtual bool BackwardBias(const _NEURO_TENSOR_DATA& current_error
					, const _VALUE_VECTOR& grad_bias){
					return true;
				}

				virtual bool SupportBackwardInnerWeight() const{ return false; }
				virtual bool BackwardInnerWeight(neuro_u32 inner_index
					, const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& input_data	// 필요없음
					, const _VALUE_VECTOR& grad_weight){
					return false;
				}

			protected:
				HiddenLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				bool AfterForward(){ return true; }

				const nsas::_LAYER_STRUCTURE_UNION& m_entry;

				_input_vector m_input_vector;
				tensor::TensorShape m_in_ts;

				activation::ActivationFunction* m_activation;

				_NEURO_TENSOR_DATA m_error_buffer;

				// layer data like weight, bias, etc.
				_layer_data_vector m_inner_data_vector;
			};

			typedef std::vector<HiddenLayerEngine*> _hidden_engine_vector;

			// HiddenLayerEngine의 상속 class로는 HiddenLayerEngine 뿐만 아니라 activation이 따로 없는 batch_normalization, dropout 등등이 있을수 있다.
			// concat은 필요 없을듯. 대신 fc connected layer을 사용하면 되지 않을까???
		}
	}
}
