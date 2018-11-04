#pragma once

#include "ConcatLayerEngine.h"

namespace np
{
	namespace engine
	{
		// ������ layer �� �ϴ� ���� ���⿡���� �� ���� �׷��� layer ���� �����ϵ��� �ؼ�
		// �ϳ��� layer�� �Ѱ� �̻��� layer�� ������, �� layer ���� ������ layer ���� �κ� ������ �� �� �ֵ��� �Ѵ�.
		namespace layers
		{
			class ConcatLayerCudaEngine : public ConcatLayerEngine
			{
			public:
				ConcatLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~ConcatLayerCudaEngine();

			protected:
				bool Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output_buffer) override;
				bool Backward(neuro_u32 batch_size) override;
			};
		}
	}
}
