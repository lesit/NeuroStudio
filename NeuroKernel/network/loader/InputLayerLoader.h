#pragma once

#include "../NeuralNetwork.h"
#include "../../nsas/NeuroEntryAccess.h"

namespace np
{
	namespace network
	{
		namespace loader
		{
			class InputLayerLoader
			{
			public:
				InputLayerLoader(nsas::NetworkEntriesReader& reader, util::UniqueIdFactory& idFactory
					, std::unordered_set<InputLayer*>& input_set
					, _uid_layer_map& layer_map)
					: m_reader(reader), m_id_factory(idFactory), m_input_layer_set(input_set), m_layer_map(layer_map)
				{
				}

				_LINKED_LAYER LoadLayer()
				{
					_LINKED_LAYER linked_layer;
					memset(&linked_layer, 0, sizeof(_LINKED_LAYER));

					neuro_size_t n = m_reader.GetInputLayerCount();
					for (neuro_size_t i = 0; i < n; i++)
					{
						nsas::_INPUT_LAYER_ENTRY entry;
						if (!m_reader.ReadInputLayer(entry))
						{
							DEBUG_OUTPUT(L"failed read input layer[%u]\r\n", i);
							continue;
						}

						if (m_layer_map.find(entry.uid) != m_layer_map.end())
						{
							DEBUG_OUTPUT(L"there is already the layer[%u]", entry.uid);
							continue;
						}

						if (!m_id_factory.InsertId(entry.uid))
						{
							DEBUG_OUTPUT(L"%u failed insert uid[%u]\r\n", i, entry.uid);
							continue;
						}

						tensor::TensorShape tensor;
						entry.ts.Get(tensor);

						InputLayer* layer = new InputLayer(entry.uid, tensor);
						if (layer)
						{
							m_input_layer_set.insert(layer);

							m_layer_map.insert(_uid_layer_map::value_type(entry.uid, layer));

							++linked_layer.count;
							if (linked_layer.end)
							{
								linked_layer.end->SetNext(layer);
								layer->SetPrev(linked_layer.end);
							}

							if (linked_layer.start == NULL)
								linked_layer.start = layer;
							linked_layer.end = layer;
						}
						else
						{
							DEBUG_OUTPUT(L"failed create instance[%u]\r\n", i);
							break;
						}
					}
					return linked_layer;
				}

			private:
				nsas::NetworkEntriesReader& m_reader;
				util::UniqueIdFactory& m_id_factory;

				std::unordered_set<InputLayer*>& m_input_layer_set;

				_uid_layer_map& m_layer_map;
			};
		}
	}
}

