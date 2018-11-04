#include "stdafx.h"

#include "WeightStoreManager.h"

#include "layers/HiddenLayerEngine.h"

#include "LayerWeightInit.h"
#include "../nsas/NeuroEntryAccess.h"

using namespace np::engine;

WeightStoreManager::WeightStoreManager(core::math_device_type pdtype, nsas::NeuroStorageAllocationSystem& nsas)
: m_pdtype(pdtype), m_nsas(nsas), m_temp_rw_buffer(core::math_device_type::cpu, true)
{
}

WeightStoreManager::~WeightStoreManager()
{
}

inline _VALUE_VECTOR WeightStoreManager::GetAccessBuffer(const _VALUE_VECTOR& buffer)
{
	if (buffer.mm.GetType() == core::math_device_type::cpu)
		return buffer;

	if (m_temp_rw_buffer.Alloc(buffer.count) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc temp buffer");
		return _VALUE_VECTOR();
	}
	_VALUE_VECTOR access_buffer = m_temp_rw_buffer;
	access_buffer.count = buffer.count;
	return access_buffer;
}

inline bool WeightStoreManager::ReadWeights(const neuro_u32 nid, neuro_u32 start, const _VALUE_VECTOR& data)
{
	if (data.count == 0)
	{
		DEBUG_OUTPUT(L"weight buffer size is zero");
		return true;
	}

	nsas::NeuroDataNodeAccessor nda_accessor(m_nsas, nid, false);
	nsas::NeuroDataAccessManager* nda = nda_accessor.GetReadAccessor();
	if (nda == NULL)
	{
		DEBUG_OUTPUT(L"no read nid[%u] accessor", nid);
		return false;
	}

	if ((start + data.count) * sizeof(neuron_weight) > nda->GetSize())
	{	// 아직 초기화 안된것이다!
		data.SetZero();
		return true;
	}

	_VALUE_VECTOR read_buffer = GetAccessBuffer(data);
	if (read_buffer.buffer == NULL)
	{
		DEBUG_OUTPUT(L"failed allocate buffer to read");
		return false;
	}

	nda->SetDataPointer(start * sizeof(neuron_value));
	if (!nda->ReadData(read_buffer.buffer, sizeof(neuron_weight) * data.count))
	{
		DEBUG_OUTPUT(L"failed to read. %u", nid);
		return false;
	}

	if (data.buffer!= read_buffer.buffer)
	{
		if (!data.CopyFrom(read_buffer))
		{
			DEBUG_OUTPUT(L"failed copy weight");
			return false;
		}
	}

	return true;
}

inline bool WeightStoreManager::WriteWeights(const neuro_u32 nid, const _VALUE_VECTOR& data)
{
	if (data.count == 0)
		return true;

	_VALUE_VECTOR write_buffer = GetAccessBuffer(data);
	if (write_buffer.buffer == NULL)
	{
		DEBUG_OUTPUT(L"failed allocate buffer to read");
		return false;
	}

	if (write_buffer.buffer!=data.buffer)
	{
		if (!write_buffer.CopyFrom(data))
		{
			DEBUG_OUTPUT(L"failed copy weight buffer");
			return false;
		}
	}

	nsas::NeuroDataNodeAccessor nda_accessor(m_nsas, nid, false);
	nsas::NeuroDataAccessManager* nda = nda_accessor.GetWriteAccessor();
	if (nda == NULL)
	{
		DEBUG_OUTPUT(L"no write nid[%u] accessor", nid);
		return neuro_last32;
	}
	if (!nda->SetSize(sizeof(neuron_weight) * write_buffer.count))
	{
		DEBUG_OUTPUT(L"failed set size : %u count", write_buffer.count);
		return false;
	}

	if (!nda->WriteData(write_buffer.buffer, sizeof(neuron_weight) * write_buffer.count))
	{
		DEBUG_OUTPUT(L"failed write");
		return false;
	}
	return true;
}

bool WeightStoreManager::ReadAllWeights()
{
	const neuro_u32 count = m_layer_engine_vector.size();
	DEBUG_OUTPUT(L"start. total %llu layers", count);

	for (neuro_u32 layer=0; layer<count; layer++)
	{
		const HiddenLayerEngine* engine = m_layer_engine_vector[layer];
		const _layer_data_vector& data_vector = engine->GetInnerDataVector();

		for (neuro_u32 i = 0; i < data_vector.size(); i++)
		{
			const _LAYER_INNER_DATA& info = data_vector[i];
			if (!ReadWeights(info.nid, 0, info.data))
			{
				DEBUG_OUTPUT(L"failed ReadWeights. layer[%u]", engine->m_layer.uid);
				return false;
			}
		}
	}
	DEBUG_OUTPUT(L"end");
	return true;
}

bool WeightStoreManager::InitAllWeights(bool is_init_weight_zero, neuro_u32 history_count)
{
	const neuro_u32 count = m_layer_engine_vector.size();
	DEBUG_OUTPUT(L"start. total %llu layers", count);

	for (neuro_u32 layer = 0; layer<count; layer++)
	{
		HiddenLayerEngine* engine = m_layer_engine_vector[layer];
		_layer_data_vector& data_vector = engine->GetInnerDataVector();

		LayerWeightInit weight_init(engine->GetLayerType(), engine->GetEntry(), engine->GetInputDataShape());

		for (neuro_u32 i = 0; i < data_vector.size(); i++)
		{
			_LAYER_INNER_DATA& info = data_vector[i];
			nsas::NeuroDataNodeAccessor nda_accessor(m_nsas, info.nid, false);
			const nsas::NeuroDataAccessManager* nda = nda_accessor.GetReadAccessor();
			if (nda == NULL)
			{
				DEBUG_OUTPUT(L"no read nid[%u] accessor", info.nid);
				return false;
			}

			if (!info.InitLearnableData(info.wtype != network::_layer_data_type::other ? history_count : 0))
			{
				DEBUG_OUTPUT(L"failed initialized learnable data");
				return false;
			}
			if (info.history.count>0)
			{
				if (is_init_weight_zero || nda->GetSize() < sizeof(neuron_weight) * (info.data.count + info.history.count))
				{
					DEBUG_OUTPUT(L"init history");
					info.history.SetZero();
				}
				else
				{
					if (!ReadWeights(info.nid, info.data.count, info.history))
					{
						DEBUG_OUTPUT(L"failed ReadWeights. layer[%u]", engine->m_layer.uid);
						return false;
					}
				}
			}

			// 초기화 안됐을 경우도 포함
			if (is_init_weight_zero || nda->GetSize() < sizeof(neuron_weight) * info.data.count)
			{
				network::_weight_init_type init_type = engine->GetWeightInitType(info.wtype);
				if (init_type == network::_weight_init_type::Zero)
				{
					DEBUG_OUTPUT(L"init weights");
					info.data.SetZero();
				}
				else
				{
					_VALUE_VECTOR temp_buffer = GetAccessBuffer(info.data);

					weight_init.InitValues(init_type, info.data.count, temp_buffer.buffer);// 실제 weight만 초기화 한다.

					if (temp_buffer.buffer == info.data.buffer)
						continue;

					if (!info.data.CopyFrom(temp_buffer))
					{
						DEBUG_OUTPUT(L"failed copy initialized weight buffer");
						return false;
					}
				}
			}
		}
	}
	WeightsToSnapshot();

	DEBUG_OUTPUT(L"end");
	return true;
}

bool WeightStoreManager::WeightsToSnapshot()
{
	return ManageSnapshotWeights(true);
}

bool WeightStoreManager::SnapshotToWeights()
{
	return ManageSnapshotWeights(false);
}

inline bool WeightStoreManager::ManageSnapshotWeights(bool snapshot)
{
	const neuro_u32 count = m_layer_engine_vector.size();
	DEBUG_OUTPUT(L"start. %s. total %llu layers", snapshot ? L"snapshot weights" : L"patch snapshot weights", count);

	for (neuro_u32 layer=0;layer<count;layer++)
	{
		const HiddenLayerEngine* engine = m_layer_engine_vector[layer];
		const _layer_data_vector& data_vector = engine->GetInnerDataVector();

		for (neuro_u32 i = 0; i < data_vector.size(); i++)
		{
			const _LAYER_INNER_DATA& info = data_vector[i];

			bool bRet;
			if (snapshot)
			{
				const core::MemoryManager& target_mm = info.snapshot.mm;
				const core::MemoryManager& source_mm = info.data.mm;

				bRet = target_mm.Memcpy(info.snapshot.buffer, info.data.buffer, sizeof(neuron_weight)* info.data.count, source_mm);
				if (bRet && info.history.count>0)
					bRet = target_mm.Memcpy(info.snapshot.buffer + info.data.count, info.history.buffer
						, sizeof(neuron_weight)* info.history.count, source_mm);
			}
			else
			{
				const core::MemoryManager& target_mm = info.data.mm;
				const core::MemoryManager& source_mm = info.snapshot.mm; 
				bRet = target_mm.Memcpy(info.data.buffer, info.snapshot.buffer, sizeof(neuron_weight)* info.data.count, source_mm);
				if (bRet && info.history.count>0)
					bRet = target_mm.Memcpy(info.history.buffer, info.snapshot.buffer + info.data.count
						, sizeof(neuron_weight)* info.history.count, source_mm);

			}

			if(!bRet)
			{
				DEBUG_OUTPUT(L"failed patch snapshot layer[%u], %uth", engine->m_layer.uid, i);
				return false;
			}
		}
	}
	DEBUG_OUTPUT(L"end");
	return true;
}

bool WeightStoreManager::UpdateWeights()
{
	const neuro_u32 count = m_layer_engine_vector.size();
	DEBUG_OUTPUT(L"start. total %llu layers", count);

	for (neuro_u32 layer = 0; layer<count; layer++)
	{
		const HiddenLayerEngine* engine = m_layer_engine_vector[layer];
		const _layer_data_vector& data_vector = engine->GetInnerDataVector();

		for (neuro_u32 i = 0; i < data_vector.size(); i++)
		{
			const _LAYER_INNER_DATA& info = data_vector[i];

			if (!WriteWeights(info.nid, info.snapshot))
			{
				DEBUG_OUTPUT(L"failed to process weights. layer[%u]", engine->m_layer.uid);
				return false;
			}
		}
	}
	DEBUG_OUTPUT(L"end");
	return true;
}
