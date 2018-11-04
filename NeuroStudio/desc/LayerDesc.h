#pragma once

#include "NeuroKernel/network/HiddenLayer.h"

#include "network/NetworkMatrix.h"

#include "TensorShapeDesc.h"

namespace np
{
	namespace str_rc
	{
		class LayerDesc
		{
		public:
			static const wchar_t* GetSimpleName(const network::AbstractLayer& layer)
			{
				network::HiddenLayer& hidden = (network::HiddenLayer&)layer;
				switch (hidden.GetLayerType())
				{
				case network::_layer_type::input:
					return L"Input";
				case network::_layer_type::output:
					return L"Output";
				case network::_layer_type::fully_connected:
					return L"FC";
				case network::_layer_type::convolutional:
					return L"Conv";
				case network::_layer_type::pooling:
					switch ((network::_pooling_type)hidden.GetEntry().pooling.type)
					{
					case network::_pooling_type::max_pooling:
						return L"Max Pool";
					case network::_pooling_type::ave_pooling:
						return L"Ave Pool";
					}
					break;
				case network::_layer_type::dropout:
					return L"Dropout";
				case network::_layer_type::rnn:
					switch ((network::_rnn_type)hidden.GetEntry().rnn.type)
					{
					case network::_rnn_type::lstm:
						return L"LSTM";
						break;
					case network::_rnn_type::gru:
						return L"GRU";
						break;
					}
					break;
				case network::_layer_type::batch_norm:
					return L"BN";
					break;
				case network::_layer_type::concat:
					return L"Concat";
					break;
				}

				return L"";
			}

			static std::wstring GetDetailName(const network::AbstractLayer& layer)
			{
				network::HiddenLayer& hidden = (network::HiddenLayer&)layer;
				switch (hidden.GetLayerType())
				{
				case network::_layer_type::input:
					return L"input";
				case network::_layer_type::fully_connected:
					return L"fc";
				case network::_layer_type::convolutional:
					return L"Convolution";
				case network::_layer_type::pooling:
					switch ((network::_pooling_type)hidden.GetEntry().pooling.type)
					{
					case network::_pooling_type::max_pooling:
						return L"Max Pooling";
					case network::_pooling_type::ave_pooling:
						return L"Average Pooling";
					}
				case network::_layer_type::dropout:
					return L"Dropout";
				case network::_layer_type::rnn:
					switch ((network::_rnn_type)hidden.GetEntry().rnn.type)
					{
					case network::_rnn_type::lstm:
						return L"LSTM";
					case network::_rnn_type::gru:
						return L"GRU";
					}
				case network::_layer_type::batch_norm:
					return L"Batch Normalization";
				case network::_layer_type::concat:
					return L"Concatenation";
				}

				return L"";
			}

			static std::wstring GetDesignDesc(const network::AbstractLayer& layer)
			{
				std::wstring ret = GetSimpleName(layer);
				if (layer.GetLayerType() != network::_layer_type::input)
				{
					network::_activation_type activation = ((network::HiddenLayer&)layer).GetActivation();
					if (activation != network::_activation_type::none)
					{
						ret += L" : ";
						ret += ToString(activation);
					}
				}
				return ret;
			}

			static std::wstring GetDesignSubDesc(const network::AbstractLayer& layer)
			{
				std::wstring ret;

				if (layer.GetLayerType() == network::_layer_type::input)
				{
					const tensor::TensorShape& tensor = layer.GetOutTensorShape();
					if (tensor.time_length > 1)
						ret += util::StringUtil::Format<wchar_t>(L"time = %u\r\n", tensor.time_length);

					ret += util::StringUtil::Format<wchar_t>(L"%u x %u x %u"
						, tensor.GetChannelCount(), tensor.GetHeight(), tensor.GetWidth());
					return ret;
				}

				const nsas::_LAYER_STRUCTURE_UNION& entry = ((network::HiddenLayer&)layer).GetEntry();
				switch (layer.GetLayerType())
				{
				case network::_layer_type::fully_connected:
					ret += util::StringUtil::Format<wchar_t>(L"%u", entry.fc.output_count);
					break;
				case network::_layer_type::convolutional:
					ret += util::StringUtil::Format<wchar_t>(L"%u x %u x %u\r\n +(%u, %u)"
						, entry.conv.channel_count
						, entry.conv.filter.kernel_width, entry.conv.filter.kernel_height
						, entry.conv.filter.stride_width, entry.conv.filter.stride_height);
					break;
				case network::_layer_type::pooling:
					ret += util::StringUtil::Format<wchar_t>(L"%u x %u, +(%u, %u)"
						, entry.pooling.filter.kernel_width, entry.pooling.filter.kernel_height
						, entry.pooling.filter.stride_width, entry.pooling.filter.stride_height);
					break;
				case network::_layer_type::dropout:
					ret += util::StringUtil::Format<wchar_t>(L"%.2f %%", entry.dropout.dropout_rate * neuro_float(100));
					break;
				}

				return ret;
			}

			static std::wstring GetDisplayDesc(const network::AbstractLayer& layer, const MATRIX_POINT& mp)
			{
				std::wstring ret = GetSimpleName(layer);
				ret += L" [";
				ret += mp.ToSimpleString();
				ret += L"]\r\n";
				ret += TensorShapeDesc::GetTensorText(layer.GetOutTensorShape());
				return ret;
			}
		};
	}
}
