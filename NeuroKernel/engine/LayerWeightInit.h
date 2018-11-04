#if !defined(_LAYER_WEIGHT_INIT_H)
#define _LAYER_WEIGHT_INIT_H

#include "../network/NeuralNetworkTypes.h"
#include "nsas/NeuralNetworkEntrySpec.h"

namespace np
{
	namespace engine
	{
		class LayerWeightInit
		{
		public:
			LayerWeightInit(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry, const tensor::TensorShape& in_ts);

			void InitValues(network::_weight_init_type init_type, neuro_size_t count, neuron_weight* buffer) const;

		protected:
			neuro_size_t fan_in_size(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry, const tensor::TensorShape& in_ts) const;
			neuro_size_t fan_out_size(network::_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION& entry) const;

		private:
			neuro_size_t m_fan_in;
			neuro_size_t m_fan_out;
		};
	}
}

#endif

