#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class PoolLayerConfigure : public HiddenLayerConfigure
		{
		public:
			virtual network::_layer_type GetLayerType() const override { return network::_layer_type::pooling; }

			void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) override
			{
				if (entry.pooling.type > (neuro_u8)network::_pooling_type::ave_pooling)
					entry.pooling.type = (neuro_u8)network::_pooling_type::max_pooling;

				if (entry.pooling.filter.kernel_width == 0)
					entry.pooling.filter.kernel_width = 1;
				if (entry.pooling.filter.kernel_height == 0)
					entry.pooling.filter.kernel_height = 1;

				if (entry.pooling.filter.stride_height == 0)
					entry.pooling.filter.stride_height = 1;
				if (entry.pooling.filter.stride_width == 0)
					entry.pooling.filter.stride_width = 1;
			}

			tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const override
			{
				if (layer.GetMainInput() == NULL)
					return tensor::TensorShape();

				const nsas::_LAYER_STRUCTURE_UNION& entry = layer.GetEntry();

				tensor::TensorShape in_ts = layer.GetMainInputTs();
				if (entry.pooling.filter.kernel_height > in_ts.GetHeight())
					const_cast<neuro_16&>(entry.pooling.filter.kernel_height) = in_ts.GetHeight();

				if (entry.pooling.filter.kernel_width > in_ts.GetWidth())
					const_cast<neuro_16&>(entry.pooling.filter.kernel_width) = in_ts.GetWidth();

				if (entry.pooling.filter.kernel_height == 0 || entry.pooling.filter.kernel_width == 0)
					return tensor::TensorShape();

				const nsas::_FILTER_ENTRY& filter = entry.pooling.filter;
				return tensor::TensorShape(1, in_ts.GetChannelCount()
					, core::filter_output_length_mode(_pad_type::valid, in_ts.GetHeight(), filter.kernel_height, filter.stride_height)
					, core::filter_output_length_mode(_pad_type::valid, in_ts.GetWidth(), filter.kernel_width, filter.stride_width));
			}
		};
	}
}
