#pragma once

#include "PoolingLayerEngine.h"

namespace np
{
	namespace engine
	{
		// ������ layer �� �ϴ� ���� ���⿡���� �� ���� �׷��� layer ���� �����ϵ��� �ؼ�
		// �ϳ��� layer�� �Ѱ� �̻��� layer�� ������, �� layer ���� ������ layer ���� �κ� ������ �� �� �ֵ��� �Ѵ�.
		// pooling�� activation�� �ʿ䰡 ����!!
		// dropoutó�� activation�� ������ �ʰ� �غ���!
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
