#pragma once

#include "PoolingLayerEngine.h"

namespace np
{
	namespace engine
	{
		// 보통은 layer 로 하는 것을 여기에서는 더 작은 그룹인 layer 으로 연결하도록 해서
		// 하나의 layer에 한개 이상의 layer을 가지고, 두 layer 간의 연결은 layer 간의 부분 연결을 할 수 있도록 한다.
		// pooling은 activation할 필요가 없다!!
		// dropout처럼 activation을 통하지 않게 해보자!
		namespace layers
		{
			class PoolingLayerCpuEngine : public PoolingLayerEngine
			{
			public:
				PoolingLayerCpuEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~PoolingLayerCpuEngine();

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;

			protected:
				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;

				_TYPED_TENSOR_DATA<neuro_u32, sizeof(neuro_u32)> m_max_index_vector;
			};
		}
	}
}
