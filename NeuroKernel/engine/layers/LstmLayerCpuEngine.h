#pragma once

#include "LstmLayerEngine.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class LstmLayerCpuEngine : public LstmLayerEngine
			{
			public:
				LstmLayerCpuEngine(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const RecurrentLayerEngine* prev_conn);
				virtual ~LstmLayerCpuEngine();

				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;

				bool BackwardWeight(neuro_u32 index
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& input_data
					, const _VALUE_VECTOR& grad_weight) override;

			protected:
				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

				// 더 빠르게 하기 위해
				_VALUE_VECTOR m_hidden_to_gate;
				_VALUE_VECTOR m_hidden_to_hidden;
			};
		}
	}
}
