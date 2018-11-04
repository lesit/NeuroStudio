#pragma once

#include "ConvLayerEngineBase.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class ConvLayerEngine : public ConvLayerEngineBase
			{
			public:
				ConvLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer);
				virtual ~ConvLayerEngine();

				virtual neuro_u32 Get1MultiplierSize() const override;

			protected:
				virtual bool OnInitialized() override;

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

				bool forward_gemm(const neuron_value* input, const neuron_value* weights, neuron_value* output);
				bool forward_bias(const neuron_value* bias, neuron_value* output);
				bool backward_gemm(const neuron_value* current_error, const neuron_value* weights, neuron_value* input_error);
				bool weight_gemm(const neuron_value* input, const neuron_value* current_error, neuron_value* grad_weight);
				bool backward_bias(const neuron_value* current_error, neuron_value* grad_bias);

				neuro_u32 m_out_spatial_dim;
				neuro_u32 m_conv_out_spatial_dim;
				neuro_u32 m_kernel_dim;
				_VALUE_VECTOR m_col_buffer;
			};
		}
	}
}
