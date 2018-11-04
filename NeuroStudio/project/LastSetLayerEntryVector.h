#pragma once

#include "NeuroKernel/nsas/NeuralNetworkEntrySpec.h"
namespace np
{
	namespace project
	{
// CNeuroDataDefineDlg 대화 상자입니다.
		class LastSetLayerEntryVector : protected std::vector<nsas::_LAYER_STRUCTURE_UNION>
		{
		public:
			LastSetLayerEntryVector()
			{
				m_hidden_type = network::_layer_type::fully_connected;
				m_activation = network::_activation_type::leakyReLU;
			}
			virtual ~LastSetLayerEntryVector() {}

			const nsas::_LAYER_STRUCTURE_UNION& GetEntry(network::_layer_type type) const
			{
				const_cast<LastSetLayerEntryVector*>(this)->AdjustSize(type);
				return at((neuro_u32)type);
			}

			void SetEntry(network::_layer_type type, const nsas::_LAYER_STRUCTURE_UNION& entry)
			{
				m_hidden_type = type;
				AdjustSize(type);
				at((neuro_u32)type) = entry;
			}

			void AdjustSize(network::_layer_type type)
			{
				neuro_u32 prev_set_count = size();
				if (prev_set_count <= (neuro_u32)type)
				{
					resize((neuro_u32)type + 1);
					for (neuro_u32 i = prev_set_count; i <= (neuro_u32)type; i++)
					{
						nsas::_LAYER_STRUCTURE_UNION& entry = at(i);
						memset(&entry, 0, sizeof(nsas::_LAYER_STRUCTURE_UNION));
						switch ((network::_layer_type)i)
						{
						case network::_layer_type::fully_connected:
							entry.fc.output_count = 1;
							break;
						case network::_layer_type::convolutional:
							entry.conv.channel_count = 1;
							entry.conv.filter.kernel_width = 5;
							entry.conv.filter.kernel_height = 5;
							entry.conv.dilation_width = 1;
							entry.conv.dilation_height = 1;
							entry.conv.filter.stride_width = 1;
							entry.conv.filter.stride_height = 1;
							entry.conv.pad_type = (neuro_u8)_pad_type::valid;
							break;
						case network::_layer_type::pooling:
							entry.pooling.type = (neuro_u8)network::_pooling_type::max_pooling;
							entry.pooling.filter.kernel_height = entry.pooling.filter.kernel_width = 2;
							entry.pooling.filter.stride_height = entry.pooling.filter.stride_width = 2;
							break;
						case network::_layer_type::dropout:
							entry.dropout.dropout_rate = neuro_float(0.5);
							break;
						case network::_layer_type::rnn:
							entry.rnn.type = (neuro_u8)network::_rnn_type::lstm;
							entry.rnn.fix_time_length = 1;
							entry.rnn.output_count = 1;
							entry.rnn.is_non_time_input = 0;
							break;
						case network::_layer_type::batch_norm:
							entry.batch_norm.momentum = 0.999f;	// memontum
							entry.batch_norm.eps = 1e-05f;
							break;
						case network::_layer_type::output:
							entry.output.loss_type = (neuro_u16)network::_loss_type::SoftmaxWithLoss;
							break;
						}
					}
				}
			}

			network::_layer_type m_hidden_type;
			network::_activation_type m_activation;
		};
	}
}
