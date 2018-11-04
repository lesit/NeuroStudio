#pragma once;

#include "../NeuralNetwork.h"
#include "../../nsas/NeuroEntryAccess.h"

namespace np
{
	namespace network
	{
		namespace writer
		{
			class InputLayerWriter
			{
			public:
				InputLayerWriter(nsas::NetworkEntriesWriter& writer)
					: m_writer(writer)
				{
					layer_count = tensor_size = 0;
				}

				bool WriteLayer(const _LINKED_LAYER& layers)
				{
					layer_count = 0;

					AbstractLayer* layer = layers.start;
					while (layer)
					{
						_INPUT_LAYER_ENTRY entry;
						entry.uid = layer->uid;
						entry.ts.Set(layer->GetOutTensorShape());

						if(!m_writer.WriteInputLayer(entry))
						{
							DEBUG_OUTPUT(L"failed write input layer[%u]\r\n", layer_count);
							return false;
						}
						++layer_count;
						tensor_size += layer->GetOutTensorShape().GetTensorSize();

						if (layer == layers.end)
							break;
						layer = layer->GetNext();
					}

					return true;
				}

				neuro_u32 layer_count;
				neuro_u64 tensor_size;
			private:
				nsas::NetworkEntriesWriter& m_writer;
			};
		}
	}
}
