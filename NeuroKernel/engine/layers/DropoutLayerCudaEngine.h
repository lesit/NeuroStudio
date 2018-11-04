#pragma once

#include "DropoutLayerEngine.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class DropoutLayerCudaEngine : public DropoutLayerEngine
			{
			public:
				DropoutLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~DropoutLayerCudaEngine();

				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;
			};
		}
	}
}

