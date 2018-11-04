#include "stdafx.h"

#include "ConvLayerEngineBase.h"
#include "ConvLayerEngine.h"
#include "ConvLayerCudnnEngine.h"

using namespace np::engine;
using namespace np::engine::layers;

ConvLayerEngineBase* ConvLayerEngineBase::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	if (net_param.run_pdtype == core::math_device_type::cuda)
	{
		tensor::TensorShape input_ts = layer.GetMainInputTs();
		tensor::TensorShape output_ts = layer.GetOutTensorShape();

		const nsas::_CONVOLUTIONAL_LAYER_ENTRY& conv_entry = layer.GetEntry().conv;
		std::pair<neuro_u32, neuro_u32> pad_height = core::filter_pad((_pad_type)conv_entry.pad_type
			, input_ts.GetHeight(), conv_entry.filter.kernel_height, conv_entry.filter.stride_height, output_ts.GetHeight());
		std::pair<neuro_u32, neuro_u32> pad_width = core::filter_pad((_pad_type)conv_entry.pad_type
			, input_ts.GetWidth(), conv_entry.filter.kernel_width, conv_entry.filter.stride_width, output_ts.GetWidth());

		if (pad_height.first == pad_height.second && pad_width.first == pad_width.second)
		{
			DEBUG_OUTPUT(L"gpu mode. pad's are same. so run on cudnn mode");
			return new ConvLayerCudnnEngine(net_param, layer);
		}
	}
	return new ConvLayerEngine(net_param, layer);
}

ConvLayerEngineBase::ConvLayerEngineBase(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: HiddenLayerEngine(net_param, layer)
{
}

ConvLayerEngineBase::~ConvLayerEngineBase()
{
}

bool ConvLayerEngineBase::OnInitialized()
{
	const nsas::_CONVOLUTIONAL_LAYER_ENTRY& conv_entry = m_entry.conv;
	if (conv_entry.channel_count != m_out_ts.GetChannelCount())
	{
		DEBUG_OUTPUT(L"kernel channel count is not same with output channel count");
		return false;
	}

	if (m_inner_data_vector[0].data.count != m_in_ts.GetChannelCount()*m_out_ts.GetChannelCount()*conv_entry.filter.kernel_height*conv_entry.filter.kernel_width)
	{
		DEBUG_OUTPUT(L"weight size[%u] is not match width kernel size. in ch[%u] X out ch[%u] X kernel h[%u] X kernel w[%u]"
			, m_inner_data_vector[0].data.count, m_in_ts.GetChannelCount(), m_out_ts.GetChannelCount(), conv_entry.filter.kernel_height, conv_entry.filter.kernel_width);
		return false;
	}

	if (m_inner_data_vector[1].data.count > 0)
	{
		if (m_inner_data_vector[1].data.count != conv_entry.channel_count)
		{
			DEBUG_OUTPUT(L"bias size is not same with output channel count");
			return false;
		}
	}

	if (conv_entry.dilation_height < 1 || conv_entry.dilation_width < 1)
	{
		DEBUG_OUTPUT(L"dilation height and width must be over 0");
		return false;
	}

	m_pad_height = core::filter_pad((_pad_type)conv_entry.pad_type
		, m_in_ts.GetHeight(), conv_entry.filter.kernel_height, conv_entry.filter.stride_height, m_out_ts.GetHeight());
	m_pad_width = core::filter_pad((_pad_type)conv_entry.pad_type
		, m_in_ts.GetWidth(), conv_entry.filter.kernel_width, conv_entry.filter.stride_width, m_out_ts.GetWidth());

	if (m_out_ts.GetHeight() != core::filter_output_length(m_in_ts.GetHeight(), conv_entry.filter.kernel_height, conv_entry.filter.stride_height, conv_entry.dilation_height, m_pad_height.first, m_pad_height.second))
	{
		DEBUG_OUTPUT(L"output height is strange");
		return false;
	}
	if (m_out_ts.GetWidth() != core::filter_output_length(m_in_ts.GetWidth(), conv_entry.filter.kernel_width, conv_entry.filter.stride_width, conv_entry.dilation_width, m_pad_width.first, m_pad_width.second))
	{
		DEBUG_OUTPUT(L"output width is strange");
		return false;
	}
	return true;
}
