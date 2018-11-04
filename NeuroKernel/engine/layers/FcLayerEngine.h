#pragma once

#include "HiddenLayerEngine.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class FcLayerEngine : public HiddenLayerEngine
			{
			public:
				static FcLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				virtual ~FcLayerEngine();

				const wchar_t* GetLayerName() const override{
					return L"Fully Connected";
				}

				virtual neuro_u32 Get1MultiplierSizePerBatch() const override;

			protected:
				FcLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				virtual bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

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
