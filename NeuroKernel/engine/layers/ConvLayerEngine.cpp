#include "stdafx.h"

#include "ConvLayerEngine.h"

#include "util/cpu_parallel_for.h"

using namespace np::engine;
using namespace np::engine::layers;

ConvLayerEngine::ConvLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: ConvLayerEngineBase(net_param, layer), m_col_buffer(net_param.run_pdtype, true)
{
	m_kernel_dim = 0;
	m_conv_out_spatial_dim = 0;
}

ConvLayerEngine::~ConvLayerEngine()
{
}

bool ConvLayerEngine::OnInitialized()
{
	if (!__super::OnInitialized())
		return false;

	m_out_spatial_dim = m_out_ts.GetHeight() * m_out_ts.GetWidth();
	m_kernel_dim = m_in_ts.GetChannelCount() * m_entry.conv.filter.kernel_height * m_entry.conv.filter.kernel_width;
	m_conv_out_spatial_dim = !reverse_dimensions() ? m_out_spatial_dim : m_in_ts.GetWidth() * m_in_ts.GetHeight();

	m_col_buffer.Alloc(m_kernel_dim * m_out_spatial_dim);
	return true;
}

neuro_u32 ConvLayerEngine::Get1MultiplierSize() const
{
	return m_out_spatial_dim;
}

bool ConvLayerEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	neuro_u32 batch_size = output_data.GetBatchTimeSize();

	const neuron_value *weight_ptr = m_inner_data_vector[0].data.buffer;
	const neuron_value *bias_ptr = m_inner_data_vector[1].data.buffer;

	for (neuro_u32 sample = 0; sample < batch_size; ++sample)
	{
		neuron_value* out_ptr = output_data.GetBatchTimeData(sample);

		if (!forward_gemm(input_data.GetBatchTimeData(sample), weight_ptr, out_ptr))
		{
			DEBUG_OUTPUT(L"failed forward_gemm");
			return false;
		}

		if (bias_ptr)
		{
			if (!forward_bias(bias_ptr, out_ptr))
			{
				DEBUG_OUTPUT(L"failed forward_bias");
				return false;
			}
		}
	}
	return true;
}

bool ConvLayerEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	neuro_u32 batch_size = current_error.GetBatchTimeSize();

	const neuron_value *weight_ptr = m_inner_data_vector[0].data.buffer;
	for (neuro_u32 sample = 0; sample < batch_size; ++sample)
	{
		if (!backward_gemm(current_error.GetBatchTimeData(sample), weight_ptr, input_error.GetBatchTimeData(sample)))
		{
			DEBUG_OUTPUT(L"failed backward_gemm");
			return false;
		}
	}

	return true;
}

bool ConvLayerEngine::BackwardWeight(neuro_u32 index
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& input_data
	, const _VALUE_VECTOR& grad_weight)
{
	neuro_u32 batch_size = current_error.GetBatchTimeSize();
	const neuro_size_t out_channel_count = m_out_ts.GetChannelCount();
	if (index == 0)
	{
		for (neuro_u32 sample = 0; sample < batch_size; ++sample)
		{
			if (!weight_gemm(input_data.GetBatchTimeData(sample), current_error.GetBatchTimeData(sample), grad_weight.buffer))
			{
				DEBUG_OUTPUT(L"failed weight_gemm");
				return false;
			}
		}
	}
	else
	{
		for (neuro_u32 sample = 0; sample < batch_size; ++sample)
		{
			if (!backward_bias(current_error.GetBatchTimeData(sample), grad_weight.buffer))
			{
				DEBUG_OUTPUT(L"failed backward_bias");
				return false;
			}
		}
	}
	return true;
}

bool ConvLayerEngine::forward_gemm(const neuron_value* input, const neuron_value* weights, neuron_value* output) 
{
	if (!m_net_param.math.im2col(input, m_in_ts.GetChannelCount(),
		m_in_ts.GetHeight(), m_in_ts.GetWidth(),
		m_entry.conv.filter.kernel_height, m_entry.conv.filter.kernel_width,
		m_pad_height.first, m_pad_height.second, m_pad_width.first, m_pad_width.second,
		m_entry.conv.filter.stride_height, m_entry.conv.filter.stride_width,
		m_entry.conv.dilation_height, m_entry.conv.dilation_width,
		m_col_buffer.buffer))
	{
		DEBUG_OUTPUT(L"failed im2col");
		return false;
	}


	if (!m_net_param.math.gemm(CblasNoTrans, CblasNoTrans
		, m_out_ts.GetChannelCount()
		, m_conv_out_spatial_dim
		, m_kernel_dim
		, (neuron_value)1., weights, m_col_buffer.buffer
		, (neuron_value)0., output))
	{
		DEBUG_OUTPUT(L"failed gemm");
		return false;
	}
	return true;
}

bool ConvLayerEngine::forward_bias(const neuron_value* bias, neuron_value* output)
{
	return m_net_param.math.gemm(CblasNoTrans, CblasNoTrans
		, m_out_ts.GetChannelCount()
		, m_out_spatial_dim
		, 1
		, (neuron_value)1., bias, m_net_param.sdb.one_set_vector.buffer
		, (neuron_value)1., output);
}

bool ConvLayerEngine::backward_gemm(const neuron_value* current_error, const neuron_value* weights, neuron_value* input_error)
{
	if(!m_net_param.math.gemm(CblasTrans, CblasNoTrans
		, m_kernel_dim
		, m_conv_out_spatial_dim
		, m_out_ts.GetChannelCount()
		, (neuron_value)1., weights, current_error
		, (neuron_value)0., m_col_buffer.buffer))
	{
		DEBUG_OUTPUT(L"failed gemm");
		return false;
	}

	if(!m_net_param.math.col2im(m_col_buffer.buffer, m_in_ts.GetChannelCount(),
		m_in_ts.GetHeight(), m_in_ts.GetWidth(),
		m_entry.conv.filter.kernel_height, m_entry.conv.filter.kernel_width,
		m_pad_height.first, m_pad_height.second, m_pad_width.first, m_pad_width.second,
		m_entry.conv.filter.stride_height, m_entry.conv.filter.stride_width,
		m_entry.conv.dilation_height, m_entry.conv.dilation_width,
		input_error))
	{
		DEBUG_OUTPUT(L"failed col2im");
		return false;
	}
	return true;
}

bool ConvLayerEngine::weight_gemm(const neuron_value* input, const neuron_value* current_error, neuron_value* grad_weight)
{
	if(!m_net_param.math.im2col(input, m_in_ts.GetChannelCount(),
		m_in_ts.GetHeight(), m_in_ts.GetWidth(),
		m_entry.conv.filter.kernel_height, m_entry.conv.filter.kernel_width,
		m_pad_height.first, m_pad_height.second, m_pad_width.first, m_pad_width.second,
		m_entry.conv.filter.stride_height, m_entry.conv.filter.stride_width,
		m_entry.conv.dilation_height, m_entry.conv.dilation_width,
		m_col_buffer.buffer))
	{
		DEBUG_OUTPUT(L"failed im2col");
		return false;
	}

	if(!m_net_param.math.gemm(CblasNoTrans, CblasTrans
		, m_out_ts.GetChannelCount()
		, m_kernel_dim
		, m_conv_out_spatial_dim
		, (neuron_value)1., current_error, m_col_buffer.buffer
		, (neuron_value)1., grad_weight))
	{
		DEBUG_OUTPUT(L"failed gemm");
		return false;
	}
	return true;
}

bool ConvLayerEngine::backward_bias(const neuron_value* current_error, neuron_value* grad_bias)
{
	return m_net_param.math.gemv(CblasNoTrans
		, m_out_ts.GetChannelCount()
		, m_out_spatial_dim
		, 1., current_error, m_net_param.sdb.one_set_vector.buffer
		, 1., grad_bias);
}
