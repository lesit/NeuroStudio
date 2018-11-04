#pragma once;

#include "../NeuralNetwork.h"
#include "../../nsas/NeuroEntryAccess.h"

using namespace np::nsas;

namespace np
{
	namespace network
	{
		class HiddenLayer;
		namespace writer
		{
			// HiddenLayer에는 ExistLayerNormalNode가 절대 있을 수 없다.
			class HiddenLayerWriter
			{
			public:
				HiddenLayerWriter(nsas::NetworkEntriesReader* reader, nsas::NetworkEntriesWriter& writer);
				virtual ~HiddenLayerWriter();

				virtual bool Save(const _LINKED_LAYER& layers, bool apply_nsas_to_layer);

				neuro_u32 layer_count;
				neuro_u32 tensor_size;
			private:
				bool SaveLayer(const HiddenLayer& layer, nsas::_LAYER_DATA_NID_SET* new_nid_set);

				neuro_u32 ConfigureInputEntries(const network::_slice_input_vector& input_vector, _INPUT_ENTRY& first_input, neuro_u32& add_input_nid);

				bool ConfigureWeights(neuro_u32 layer_data_count, _LAYER_DATA_NIDS& data_nids);
				bool CopyWeights(const _LAYER_DATA_NIDS& stored_nids, _LAYER_DATA_NIDS& data_nids);

				nsas::NetworkEntriesReader* m_reader;
				nsas::NetworkEntriesWriter& m_writer;

				_VALUE_VECTOR m_write_buffer;
			};
		}
	}
}
