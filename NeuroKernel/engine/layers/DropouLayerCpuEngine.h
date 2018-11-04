#pragma once

#include "DropoutLayerEngine.h"

namespace np
{
	namespace engine
	{
		// ������ layer �� �ϴ� ���� ���⿡���� �� ���� �׷��� layer ���� �����ϵ��� �ؼ�
		// �ϳ��� layer�� �Ѱ� �̻��� layer�� ������, �� layer ���� ������ layer ���� �κ� ������ �� �� �ֵ��� �Ѵ�.
		namespace layers
		{
			class DropouLayerCpuEngine : public DropoutLayerEngine
			{
			public:
				DropouLayerCpuEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~DropouLayerCpuEngine();

				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;
			};
		}
	}
}

