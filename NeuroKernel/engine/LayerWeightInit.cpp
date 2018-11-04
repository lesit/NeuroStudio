#include "stdafx.h"
#include "LayerWeightInit.h"

#include "backend/weight_init.h"

using namespace np::engine;
using namespace np::engine::weight_init;

namespace np
{
	namespace engine
	{
		namespace weight_init
		{
			class WeightInitManager
			{
			public:
				WeightInitManager()
				{
					m_weightInit = new weight_init::Xavier;
					m_biasInit = new weight_init::Constant;
				}
				virtual ~WeightInitManager()
				{
					delete m_weightInit;
					delete m_biasInit;
				}

				weight_init::WeightInit* m_weightInit;
				weight_init::WeightInit* m_biasInit;
			};
		}
	}
}

WeightInitManager weightInitManager;

LayerWeightInit::LayerWeightInit(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry, const tensor::TensorShape& in_ts)
{
	m_fan_in = fan_in_size(layer_type, entry, in_ts);
	m_fan_out = fan_out_size(layer_type, entry);
}

void LayerWeightInit::InitValues(network::_weight_init_type init_type, neuro_size_t count, neuron_weight* buffer) const
{
	if (init_type==network::_weight_init_type::Zero)
	{
		memset(buffer, 0, sizeof(neuron_weight)*count);
	}
	else
	{
		weight_init::Constant Constant;
		weight_init::Gaussian Gaussian;
		weight_init::He He;
		weight_init::LeCun LeCun;
		weight_init::Xavier Xavier;
		WeightInit* init_ojb = &Xavier;
		switch (init_type)
		{
		case network::_weight_init_type::Constant:
			init_ojb = &Constant;
			break;
		case network::_weight_init_type::Gaussian:
			init_ojb = &Gaussian;
			break;
		case network::_weight_init_type::He:
			init_ojb = &He;
			break;
		case network::_weight_init_type::LeCun:
			init_ojb = &LeCun;
			break;
		case network::_weight_init_type::Xavier:
			init_ojb = &Xavier;
			break;
		}

		init_ojb->fill(count, buffer, m_fan_in, m_fan_out);
	}
}

neuro_size_t LayerWeightInit::fan_in_size(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry, const tensor::TensorShape& in_ts) const
{
	if (layer_type == network::_layer_type::fully_connected)
		return in_ts.GetTensorSize();
	else if (layer_type == network::_layer_type::convolutional)
		return in_ts.GetChannelCount() * entry.conv.filter.kernel_height * entry.conv.filter.kernel_width;
	else if (layer_type == network::_layer_type::rnn)
		return in_ts.GetDimSize();

	return 0;
}

neuro_size_t LayerWeightInit::fan_out_size(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry) const
{
	if (layer_type == network::_layer_type::fully_connected)
		return entry.fc.output_count;
	else if (layer_type == network::_layer_type::convolutional)
		return entry.conv.channel_count*entry.conv.filter.kernel_height*entry.conv.filter.kernel_width;
	else if (layer_type == network::_layer_type::rnn)
		return entry.rnn.output_count;

	return 0;
}
