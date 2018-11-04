#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class RnnLayerConfigure : public HiddenLayerConfigure
		{
		public:
			network::_layer_type GetLayerType() const override { return network::_layer_type::rnn; }
			virtual bool AvailableSetSideInput(const HiddenLayer& layer, const HiddenLayer* input) const override;
			virtual tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const override;
			virtual neuro_u32 GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const override;

			neuro_u32 GetGateCount(const nsas::_LAYER_STRUCTURE_UNION& entry) const;

			bool HasWeight() const override { return true; }
		};
	}
}
