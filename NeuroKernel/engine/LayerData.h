#pragma once

#include "../network/NeuralNetworkTypes.h"

namespace np
{
	namespace engine
	{
		struct _LAYER_INNER_DATA
		{
			network::_layer_data_type wtype;

			neuro_u32 nid;

			_VALUE_VECTOR data;

			_VALUE_VECTOR history;
			_VALUE_VECTOR snapshot;

			bool Alloc(core::math_device_type pdtype, neuro_u32 size)
			{
				data.Dealloc();
				data = _VALUE_VECTOR(pdtype);
				if (data.Alloc(size) == NULL)
					return false;

				return true;
			}

			bool InitLearnableData(neuro_u32 history_count)
			{
				history.Dealloc();
				snapshot.Dealloc();
				history = _VALUE_VECTOR(data.mm.GetType());
				snapshot = _VALUE_VECTOR(data.mm.GetType());
				if (data.count == 0)
					return false;

				if (history_count)
				{
					if (history.Alloc(data.count * history_count) == NULL)
						return false;
				}

				if (snapshot.Alloc(data.count*(1 + history_count)) == NULL)
					return false;
				return true;
			}

			neuron_weight* GetHistory(neuro_u32 index) const
			{
				// 2개인 이유는 history는 weight 다음에 붙기 때문
				if (history.count < data.count * (1 + index))
					return NULL;

				return history.buffer + data.count * index;
			}

			void Dealloc()
			{
				data.Dealloc();
				history.Dealloc();
				snapshot.Dealloc();
			}
		};
		typedef std::vector<_LAYER_INNER_DATA> _layer_data_vector;
	}
}
