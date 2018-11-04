#if !defined(_NETWORK_ST_WRITER_H)
#define _NETWORK_ST_WRITER_H

#include "../NeuralNetwork.h"
#include "../nsas/NeuralNetworkEntrySpec.h"

#include "InputLayerWriter.h"
#include "HiddenLayerWriter.h"

namespace np
{
	namespace network
	{
		namespace writer
		{
			class NeuralNetworkWriter
			{
			public:
				NeuralNetworkWriter(const NeuralNetwork& network)
					: m_network(network)
				{
				}

				virtual ~NeuralNetworkWriter()
				{
				}

				bool Save(NeuroStorageAllocationSystem* load_nsas, NeuroStorageAllocationSystem& save_nsas, bool apply_nsas_to_layer)
				{
					NetworkEntriesWriter writer(save_nsas);

					nsas::_NEURO_ROOT_ENTRY root_entry;

					bool is_save_as = load_nsas != &save_nsas;
					if (is_save_as)	// 새로 저장
					{
						memset(&root_entry, 0, sizeof(_NEURO_ROOT_ENTRY));

						if (load_nsas)
						{
							root_entry.opt_params = load_nsas->GetRootEntry().opt_params;
							root_entry.history = load_nsas->GetRootEntry().history;
						}
					}
					else			
					{// 기존것을 수정하는 경우
						if(!writer.LayersDeleted(m_network.GetDeletedLayerDataNidVector()))
						{
							DEBUG_OUTPUT(L"failed RemoveDeletedStoreLayers");
							return false;
						}
						memcpy(&root_entry, &load_nsas->GetRootEntry(), sizeof(_NEURO_ROOT_ENTRY));
					}

					{
						const network::_LEARNING_INFO& info = m_network.GetLearningInfo();
						root_entry.trainer.optimizer_type = (neuro_u16)info.optimizer_type;
						root_entry.trainer.lr_policy.type = (neuro_u16)info.optimizing_rule.lr_policy.type;
						root_entry.trainer.lr_policy.lr_base = info.optimizing_rule.lr_policy.lr_base;
						root_entry.trainer.lr_policy.gamma = info.optimizing_rule.lr_policy.gamma;
						root_entry.trainer.lr_policy.step = (neuro_u16)info.optimizing_rule.lr_policy.step;
						root_entry.trainer.lr_policy.power = info.optimizing_rule.lr_policy.power;

						root_entry.trainer.wn_policy.type = (neuro_u16)info.optimizing_rule.wn_policy.type;
						root_entry.trainer.wn_policy.weight_decay = info.optimizing_rule.wn_policy.weight_decay;

						root_entry.trainer.data_batch_type = (neuro_u16)info.data_batch_type;
					}

					nsas::_NEURAL_NETWORK_INFO& nn_info = root_entry.network;
					nn_info.input_layer_count = 0;
					nn_info.input_tensor_size = 0;
					nn_info.hidden_layer_count = 0;
					nn_info.hidden_tensor_size = 0;

					writer::InputLayerWriter inputLayerWriter(writer);
					if (inputLayerWriter.WriteLayer(m_network.GetInputLayers()))
					{
						nn_info.input_layer_count = inputLayerWriter.layer_count;
						nn_info.input_tensor_size = inputLayerWriter.tensor_size;
					}
					else
						DEBUG_OUTPUT(L"failed Save input layers");

					nsas::NetworkEntriesReader* reader = NULL;
					if (load_nsas !=NULL && is_save_as)
						reader = new nsas::NetworkEntriesReader(*load_nsas);

					writer::HiddenLayerWriter hiddenLayerwriter(reader, writer);
					bool ret = hiddenLayerwriter.Save(m_network.GetHiddenLayers(), apply_nsas_to_layer);

					delete reader;

					if(ret)
					{
						nn_info.hidden_layer_count = hiddenLayerwriter.layer_count;
						nn_info.hidden_tensor_size = hiddenLayerwriter.tensor_size;
					}
					else
					{
						DEBUG_OUTPUT(L"failed Save hidden layers");
						return false;
					}

					save_nsas.ChangeRootEntry(root_entry);
					save_nsas.CompleteUpdate();

					DEBUG_OUTPUT(L"input layer=%u, input tensor=%llu, hidden layer=%u, hidden tensor=%llu",
						nn_info.input_layer_count,
						nn_info.input_tensor_size,
						nn_info.hidden_layer_count,
						nn_info.hidden_tensor_size);

					return true;
				}

			private:
				const NeuralNetwork& m_network;
			};
		}
	}
}
#endif
