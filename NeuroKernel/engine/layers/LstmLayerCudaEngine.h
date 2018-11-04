#pragma once

#include "LstmLayerEngine.h"

#ifdef _DEBUG
#include "LstmLayerCpuEngine.h"
#endif

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class LstmLayerCudaEngine : public LstmLayerEngine
			{
			public:
				LstmLayerCudaEngine(const NetworkParameter& net_param
					, const network::HiddenLayer& layer
					, const RecurrentLayerEngine* prev_conn);
				virtual ~LstmLayerCudaEngine();

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
			};
		}
	}
}
