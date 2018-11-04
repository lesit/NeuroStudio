#pragma once

#include "../NeuralNetwork.h"
#include "../HiddenLayer.h"
#include "../../nsas/NeuroEntryAccess.h"

namespace np
{
	namespace network
	{
		namespace loader
		{
			class HiddenLayerLoader
			{
			public:
				HiddenLayerLoader(nsas::NetworkEntriesReader& reader, util::UniqueIdFactory& idFactory
					, std::unordered_set<OutputLayer*>& output_layer_set
					, _uid_layer_map& layer_map)
					: m_reader(reader), m_id_factory(idFactory), m_output_layer_set(output_layer_set), m_layer_map(layer_map)
				{
				}

				_LINKED_LAYER LoadLayer()
				{
					_LINKED_LAYER linked_layer;
					memset(&linked_layer, 0, sizeof(_LINKED_LAYER));

					for (neuro_u32 index = 0, n = m_reader.GetHiddenLayerCount(); index < n; index++)
					{
						nsas::_HIDDEN_LAYER_ENTRY entry;
						if (!m_reader.ReadHiddenLayer(entry))
						{
							DEBUG_OUTPUT(L"failed read %u th hidden layer", index);
							break;
						}

						if (m_layer_map.find(entry.uid) != m_layer_map.end())
						{
							DEBUG_OUTPUT(L"%u there is already the layer[%u]", index, entry.uid);
							continue;
						}

						if (!m_id_factory.InsertId(entry.uid))
						{
							DEBUG_OUTPUT(L"%u failed insert uid[%u]\r\n", index, entry.uid);
							continue;
						}

#ifdef _DEBUG
						if ((np::network::_layer_type)entry.type == network::_layer_type::output)
							int a = 0;
#endif

						HiddenLayer* layer = HiddenLayer::CreateInstance((np::network::_layer_type)entry.type, entry.uid);
						if (!layer)
						{
							DEBUG_OUTPUT(L"%u failed create instance. uid[%u]\r\n", index, entry.uid);
							m_id_factory.RemoveId(entry.uid);
							continue;
						}

						nsas::_input_entry_vector input_entry_vector;
						if (entry.input.uid != neuro_last32)
						{
							input_entry_vector.push_back(entry.input);

							if (entry.sub_nid_set.add_input_nid != neuro_last32)
							{
								NeuroDataNodeAccessor ndna(m_reader.GetNSAS(), entry.sub_nid_set.add_input_nid, true);
								const NeuroDataAccessManager* nda = ndna.GetReadAccessor();
								if (nda != NULL)
								{
									for (neuro_u32 i = 0, n = nda->GetSize() / sizeof(_INPUT_ENTRY); i < n; i++)
									{
										_INPUT_ENTRY input;
										if (!nda->ReadData(&input, sizeof(_INPUT_ENTRY)))
											break;

										input_entry_vector.push_back(input);
									}
								}
							}
						}

						layer->AttachStoredInfo(entry, input_entry_vector, m_layer_map);

						if (layer->GetLayerType() == _layer_type::output)
							m_output_layer_set.insert((OutputLayer*)layer);

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
					return linked_layer;
				}

			private:
				NetworkEntriesReader& m_reader;
				util::UniqueIdFactory& m_id_factory;

				std::unordered_set<OutputLayer*>& m_output_layer_set;

				_uid_layer_map& m_layer_map;
			};
		}
	}
}
