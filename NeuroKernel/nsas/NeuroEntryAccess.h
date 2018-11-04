#pragma once

#include "NeuroStorageAllocationSystem.h"
#include "NeuroDataAccessManager.h"

namespace np
{
	namespace nsas
	{
		typedef std::vector<nsas::_INPUT_ENTRY> _input_entry_vector;
		typedef std::vector<nsas::_LAYER_DATA_NID_SET> _layer_data_nid_vector;

		class NetworkEntriesAccessor
		{
		public:
			virtual ~NetworkEntriesAccessor() {}
			NeuroStorageAllocationSystem& GetNSAS() { return m_nsas; }

		protected:
			NetworkEntriesAccessor(NeuroStorageAllocationSystem& nsas);

			NeuroStorageAllocationSystem& m_nsas;
			NeuroDataAccessManager m_input_layer_nda;
			NeuroDataAccessManager m_hidden_layer_nda;
		};

		// NetworkLayerReader ?
		class NetworkEntriesReader : public NetworkEntriesAccessor
		{
		public:
			NetworkEntriesReader(NeuroStorageAllocationSystem& nsas);
			virtual ~NetworkEntriesReader() {}

			neuro_u32 GetInputLayerCount() const;
			bool ReadInputLayer(_INPUT_LAYER_ENTRY& entry);

			neuro_u32 GetHiddenLayerCount() const;
			bool ReadHiddenLayer(_HIDDEN_LAYER_ENTRY& entry);
		};

		// NetworkLayerWriter ?
		class NetworkEntriesWriter : public NetworkEntriesAccessor
		{
		public:
			NetworkEntriesWriter(NeuroStorageAllocationSystem& nsas);
			virtual ~NetworkEntriesWriter();

			bool LayersDeleted(const _layer_data_nid_vector& data_vector);

			bool WriteInputLayer(const _INPUT_LAYER_ENTRY& entry);
			bool WriteHiddenLayer(const _HIDDEN_LAYER_ENTRY& entry);

			bool AllocDataNda(neuro_u32 n, neuro_u32* nid_list);
			void DeallocDataNda(neuro_u32 n, neuro_u32* nid_list);

		private:
			_NEURAL_NETWORK_DEFINITION& m_nn_def;

			NeuroDataAccessManager m_data_nda_bitmap;
			NeuroDataAccessManager m_data_nda_table;

			neuro_u8* m_block_data;

			const neuro_u32 m_block_size;
			const neuro_u32 m_nU32PerBlock;
		};

		class NeuroDataNodeAccessor
		{
		public:
			// write mode의 중복 open 검사는 NeuroStorageAllocationSystem에서 하도록 하자!
			NeuroDataNodeAccessor(NeuroStorageAllocationSystem& nsas, neuro_u32 nid, bool is_readonly);
			virtual ~NeuroDataNodeAccessor();

			NeuroDataAccessManager* GetWriteAccessor() { return m_writing_nid != neuro_last32 ? m_data_nda : NULL; }
			NeuroDataAccessManager* GetReadAccessor() const { return m_data_nda; }
		private:
			NeuroDataAccessManager* Open(NeuroStorageAllocationSystem& nsas, neuro_u32 nid);

			NeuroDataAccessManager m_data_nda_table;

			const neuro_u32 m_writing_nid;

			NeuroDataAccessManager* m_data_nda;
			_NEURO_DATA_ALLOC_SPEC m_nda_spec;
		};
	}
}
