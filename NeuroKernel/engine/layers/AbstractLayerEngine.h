#pragma once

#include "common.h"
#include <unordered_map>
#include <unordered_set>

//#include "../../nsas/NeuralNetworkEntrySpec.h"

#include "../NetworkParameter.h"
#include "../../network/InputLayer.h"

namespace np
{
	using namespace tensor;

	namespace engine
	{
		// ������ layer �� �ϴ� ���� ���⿡���� �� ���� �׷��� layer ���� �����ϵ��� �ؼ�
		// �ϳ��� layer�� �Ѱ� �̻��� layer�� ������, �� layer ���� ������ layer ���� �κ� ������ �� �� �ֵ��� �Ѵ�.
		namespace layers
		{
			class AbstractLayerEngine
			{
			public:
				AbstractLayerEngine(const NetworkParameter& net_param, const network::AbstractLayer& layer);
				virtual ~AbstractLayerEngine();

				virtual network::_layer_type GetLayerType() const = 0;

				bool AllocOutputBuffer(core::math_device_type pdtype, neuro_u32 batch_size);
				void DeallocOutputBuffer();

				virtual bool IsHiddenLayer() const{ return false; }

				virtual bool IsOneHotResult() const{ return false; }

				const TensorShape& GetOutTensorShape() const 
				{
					return m_out_ts; 
				}
				
				virtual _ts_batch_time_order TensorBatchTimeOrder() const {
					return _ts_batch_time_order::NxTxD;
				}

				const _NEURO_TENSOR_DATA& GetOutputData() const {
					return m_output;
				}

				virtual std::pair<neuron_value, neuron_value> GetOutputScale() const 
				{
					return std::make_pair(neuron_value(-1.0), neuron_value(1.0)); 
				}

				virtual neuro_u32 GetPitchGPUMemSize() const{ return 0; }

				const network::AbstractLayer& m_layer;
				const TensorShape m_out_ts;
			protected:
				const NetworkParameter& m_net_param;

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) { return true; }
			private:
				_NEURO_TENSOR_DATA m_output;
			};

			class InputLayerEngine : public AbstractLayerEngine
			{
			public:
				InputLayerEngine(const NetworkParameter& net_param, const network::InputLayer& layer);

				network::_layer_type GetLayerType() const override { return network::_layer_type::input; }

				std::pair<neuron_value, neuron_value> GetOutputScale() const override
				{
					return std::make_pair(neuron_value(-1.0), neuron_value(1.0));
				}
			};
		}
		typedef std::vector<layers::AbstractLayerEngine*> _layer_engine_vector;
		typedef std::vector<layers::InputLayerEngine*> _input_engine_vector;
	}
}

