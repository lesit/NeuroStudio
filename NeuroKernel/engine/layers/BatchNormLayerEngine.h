#pragma once

#include "HiddenLayerEngine.h"

/*
	This source came from Caffe(https://github.com/BVLC/caffe)
	and has been modified for Neuro Studio structure
*/
namespace np
{
	namespace engine
	{
		namespace layers
		{
			class BatchNormLayerEngine : public HiddenLayerEngine
			{
			public:
				static BatchNormLayerEngine* CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				virtual ~BatchNormLayerEngine();

				const wchar_t* GetLayerName() const override{
					return L"Batch Normalization";
				}

				virtual neuro_u32 Get1MultiplierSize() const override;
				virtual neuro_u32 Get1MultiplierSizePerBatch() const override;

				virtual bool OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf) override;
			protected:
				BatchNormLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);

				virtual bool OnInitialized() override;

				bool ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data) override;

				bool BackwardError(const _NEURO_TENSOR_DATA& output_data
					, const _NEURO_TENSOR_DATA& current_error
					, const _NEURO_TENSOR_DATA& input_data
					, const _NEURO_TENSOR_DATA& input_error) override;

				neuro_u32 m_batch_size;
				neuro_u32 m_channel;
				neuro_u32 m_spatial_dim;

				_VALUE_VECTOR m_cpu_mean_div_variance;

				_VALUE_VECTOR m_mean, m_variance;

				_NEURO_TENSOR_DATA m_temp;
			};
		}
	}
}

