#pragma once

#include "PoolingLayerEngine.h"

#include "core/cuda_platform.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class PoolingLayerCudaEngine : public PoolingLayerEngine
			{
			public:
				PoolingLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~PoolingLayerCudaEngine();

				virtual bool OnInitialized() override;

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;
			protected:
				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;

				cudnnHandle_t			m_handle;
				cudnnTensorDescriptor_t m_input_desc, m_output_desc;
				cudnnPoolingDescriptor_t  m_pooling_desc;
			};
		}
	}
}
