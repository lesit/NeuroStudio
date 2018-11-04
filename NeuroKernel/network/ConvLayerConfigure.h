#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class ConvLayerConfigure : public HiddenLayerConfigure
		{
		public:
			network::_layer_type GetLayerType() const override { return network::_layer_type::convolutional; }

			bool HasActivation() const override { return true; }
			bool HasWeight() const override { return true; }

			void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) override;

			tensor::TensorShape MakeOutTensorShape(const HiddenLayer& layer) const override;
			virtual neuro_u32 GetLayerDataInfoVector(const HiddenLayer& layer, _layer_data_info_vector& info_vector) const override;
			
			static std::pair<neuro_u32, neuro_u32> GetPad(const HiddenLayer& layer, bool isHeight);
		};
	}
}

