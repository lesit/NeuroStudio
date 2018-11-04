#pragma once

#include "../NeuralNetwork.h"
#include "../../nsas/NeuralNetworkEntrySpec.h"
#include "../../nsas/NeuroEntryAccess.h"

#include "InputLayerLoader.h"
#include "HiddenLayerLoader.h"

namespace np
{
	namespace network
	{
		namespace loader
		{
			class NeuralNetworkLoader
			{
			public:
				NeuralNetworkLoader(nsas::NeuroStorageAllocationSystem& nsas
					, util::UniqueIdFactory& idFactory
					, _uid_layer_map& layer_map
					, _input_layer_set& input_set
					, _output_layer_set& output_layer_set)
					: m_nsas(nsas), m_reader(nsas), m_id_factory(idFactory)
					, m_layer_map(layer_map), m_input_layer_set(input_set), m_output_layer_set(output_layer_set)
				{
					m_layer_map.clear();
					m_input_layer_set.clear();
					m_output_layer_set.clear();
				}

				bool Load(_LEARNING_INFO& learning_info, _LINKED_LAYER& input, _LINKED_LAYER& hidden)
				{
					const nsas::_NEURO_ROOT_ENTRY& root_entry = m_nsas.GetRootEntry();
					DEBUG_OUTPUT(L"input=%u, hidden=%u",
						root_entry.network.input_layer_count,
						root_entry.network.hidden_layer_count);

					{
						memset(&learning_info, 0, sizeof(_LEARNING_INFO));
						learning_info.optimizer_type = (_optimizer_type)root_entry.trainer.optimizer_type;

						learning_info.optimizing_rule.lr_policy.type = (_lr_policy_type)root_entry.trainer.lr_policy.type;
						learning_info.optimizing_rule.lr_policy.lr_base = root_entry.trainer.lr_policy.lr_base;
						learning_info.optimizing_rule.lr_policy.gamma = root_entry.trainer.lr_policy.gamma;
						learning_info.optimizing_rule.lr_policy.step = root_entry.trainer.lr_policy.step;
						learning_info.optimizing_rule.lr_policy.power = root_entry.trainer.lr_policy.power;

						learning_info.optimizing_rule.wn_policy.type = (_wn_policy_type)root_entry.trainer.wn_policy.type;
						learning_info.optimizing_rule.wn_policy.weight_decay = root_entry.trainer.wn_policy.weight_decay;

						learning_info.data_batch_type = (_train_data_batch_type) root_entry.trainer.data_batch_type;
					}
					{
						InputLayerLoader loader(m_reader, m_id_factory, m_input_layer_set, m_layer_map);
						input = loader.LoadLayer();
					}

					{
						HiddenLayerLoader loader(m_reader, m_id_factory, m_output_layer_set, m_layer_map);
						hidden = loader.LoadLayer();
					}

					return true;
				}

			protected:
				nsas::NeuroStorageAllocationSystem& m_nsas;
				nsas::NetworkEntriesReader m_reader;

				_uid_layer_map& m_layer_map;

				util::UniqueIdFactory& m_id_factory;

				_input_layer_set& m_input_layer_set;
				_output_layer_set& m_output_layer_set;
			};
		}
	}
}
