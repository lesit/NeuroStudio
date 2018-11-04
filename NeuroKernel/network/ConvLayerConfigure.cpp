#include "stdafx.h"

#include "ConvLayerConfigure.h"
#include "HiddenLayer.h"

using namespace np;
using namespace np::network;

void ConvLayerConfigure::EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry)
{
	if (entry.conv.dilation_width == 0)
		entry.conv.dilation_width = 1;
	if (entry.conv.dilation_height == 0)
		entry.conv.dilation_height = 1;

	if (entry.conv.channel_count == 0)
		entry.conv.channel_count = 1;

	if (entry.conv.filter.kernel_width == 0)
		entry.conv.filter.kernel_width = 1;
	if (entry.conv.filter.kernel_height == 0)
		entry.conv.filter.kernel_height = 1;

	if (entry.conv.filter.stride_width == 0)
		entry.conv.filter.stride_width = 1;
	if (entry.conv.filter.stride_height == 0)
		entry.conv.filter.stride_height = 1;

	switch((_pad_type)entry.conv.pad_type)
	{
	case _pad_type::same:
	case _pad_type::valid:
		break;
	case _pad_type::user_define:
	default:
		DEBUG_OUTPUT(L"not supported. %u. changed valid mode", neuro_u32(entry.conv.pad_type));
		entry.conv.pad_type = (neuro_u8)_pad_type::valid;
	}
}

std::pair<neuro_u32, neuro_u32> ConvLayerConfigure::GetPad(const HiddenLayer& layer, bool isHeight)
{
	if (layer.GetLayerType() != _layer_type::convolutional)
		return{ 0,0 };

	const tensor::TensorShape& in_ts = layer.GetMainInputTs();
	ConvLayerConfigure config;
	const tensor::TensorShape& out_ts = config.MakeOutTensorShape(layer);

	const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();
	if (isHeight)
		return core::filter_pad((_pad_type)entry.conv.pad_type
		, in_ts.GetHeight(), entry.conv.filter.kernel_height, entry.conv.filter.stride_height, out_ts.GetHeight());
	else
		return core::filter_pad((_pad_type)entry.conv.pad_type
		, in_ts.GetWidth(), entry.conv.filter.kernel_width, entry.conv.filter.stride_width, out_ts.GetWidth());
}

tensor::TensorShape ConvLayerConfigure::MakeOutTensorShape(const HiddenLayer& layer) const
{
	if (layer.GetMainInput()==NULL)
		return tensor::TensorShape();

	const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

	tensor::TensorShape in_ts = layer.GetMainInputTs();
	if (entry.conv.filter.kernel_height > in_ts.GetHeight())
		const_cast<neuro_16&>(entry.conv.filter.kernel_height) = in_ts.GetHeight();

	if (entry.conv.filter.kernel_width > in_ts.GetWidth())
		const_cast<neuro_16&>(entry.conv.filter.kernel_width) = in_ts.GetWidth();

	const nsas::_FILTER_ENTRY& filter = entry.conv.filter;
	return tensor::TensorShape(1, entry.conv.channel_count
		, core::filter_output_length_mode((_pad_type)entry.conv.pad_type, in_ts.GetHeight(), filter.kernel_height, filter.stride_height, entry.conv.dilation_height)
		, core::filter_output_length_mode((_pad_type)entry.conv.pad_type, in_ts.GetWidth(), filter.kernel_width, filter.stride_width, entry.conv.dilation_width));
}

neuro_u32 ConvLayerConfigure::GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const
{
	const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

	info_vector.resize(2);
	// out channel, in channel, kernel height, kernel width
	info_vector[0] = { _layer_data_type::weight, entry.conv.channel_count * layer.GetMainInputTs().GetChannelCount() * (neuro_u32)entry.conv.filter.kernel_height * (neuro_u32)entry.conv.filter.kernel_width };
	info_vector[1] = { _layer_data_type::bias, entry.conv.channel_count };

	return info_vector.size();
}
