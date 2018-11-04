#pragma once

#include "HiddenLayerConfigure.h"

namespace np
{
	namespace network
	{
		class DropoutLayerConfigure : public HiddenLayerConfigure
		{
		public:
			virtual network::_layer_type GetLayerType() const { return network::_layer_type::dropout; }

			void EntryValidation(nsas::_LAYER_STRUCTURE_UNION& entry) override
			{
				if (entry.dropout.dropout_rate < neuro_float(0.1))
					entry.dropout.dropout_rate = neuro_float(0.1);
				else if (entry.dropout.dropout_rate > neuro_float(0.9))
					entry.dropout.dropout_rate = neuro_float(0.9);
			}
		};
	}
}
