#include "stdafx.h"

#include "HiddenLayerWriter.h"

#include "util/StringUtil.h"

#include "../HiddenLayer.h"

using namespace np;
using namespace np::network;
using namespace np::network::writer;

HiddenLayerWriter::HiddenLayerWriter(nsas::NetworkEntriesReader* reader, nsas::NetworkEntriesWriter& writer)
: m_writer(writer), m_write_buffer(core::math_device_type::cpu, true)
{
	layer_count = tensor_size = 0;
	m_reader = reader;	// reader가 있다는 것은 save as란 얘기이다.
}

HiddenLayerWriter::~HiddenLayerWriter()
{
}

bool HiddenLayerWriter::Save(const _LINKED_LAYER& layers, bool apply_nsas_to_layer)
{
	DEBUG_OUTPUT(L"start");

	typedef std::vector<nsas::_LAYER_DATA_NID_SET> _layer_nid_vector;
	_layer_nid_vector layer_nid_vector;
	if(apply_nsas_to_layer)
		layer_nid_vector.resize(layers.count);

	AbstractLayer* layer = layers.start;
	while (layer)
	{
		if (layer->GetLayerType() == _layer_type::input)
		{
			DEBUG_OUTPUT(L"the %uth layer[%u] is not hidden", layer_count, layer->uid);
			return false;
		}

		if (!SaveLayer(*(const HiddenLayer*)layer, apply_nsas_to_layer ? &layer_nid_vector[layer_count] : NULL))
		{
			DEBUG_OUTPUT(L"failed to get %uth layer[%u] info", layer_count, layer->uid);
			return false;
		}
		++layer_count;
		tensor_size += layer->GetOutTensorShape().GetTensorSize();

		if (layer == layers.end)
			break;
		layer = layer->GetNext();
	}
	if (apply_nsas_to_layer)
	{
		// 최신 nid set을 적용시켜 모두 로드된 상태로 만든다.
		DEBUG_OUTPUT(L"set layer new nid set.\r\n");

		layer = layers.start;
		for (neuro_u32 i = 0; i<layer_count; i++, layer = layer->GetNext())
			((HiddenLayer*)layer)->SetStoredNidSet(&layer_nid_vector[i]);
	}

	DEBUG_OUTPUT(L"completed.\r\n");
	return true;
}

void SetWeightInfo(const _LAYER_WEIGHT_INFO& source, nsas::_WEIGHT_INFO& target)
{
	target.init_type = (neuro_u16) source.init_type;
	target.init_scale = source.init_scale;
	target.mult_lr = source.mult_lr;
	target.decay = source.decay;
}

bool HiddenLayerWriter::SaveLayer(const HiddenLayer& layer, nsas::_LAYER_DATA_NID_SET* new_nid_set)
{
	_HIDDEN_LAYER_ENTRY entry;
	memset(&entry, 0, sizeof(_HIDDEN_LAYER_ENTRY));

	entry.uid = layer.uid;
	entry.type = (neuro_u16)layer.GetLayerType();
	entry.basic_info.activation = (neuro_u16)layer.GetActivation();
	const _LAYER_WEIGHT_INFO* weight_info = layer.GetWeightInfo(_layer_data_type::weight);
	const _LAYER_WEIGHT_INFO* bias_info = layer.GetWeightInfo(_layer_data_type::bias);
	if (weight_info)
		SetWeightInfo(*weight_info, entry.basic_info.weight_info);
	if (bias_info)
		SetWeightInfo(*bias_info, entry.basic_info.bias_info);

	entry.function = layer.GetEntry();
	entry.sub_nid_set.add_input_nid = neuro_last32;

	// save as가 아닐때 기존 nid set 복사
	if (m_reader==NULL)
		entry.sub_nid_set = layer.GetStoredNidSet();

	if (ConfigureInputEntries(layer.GetInputVector(), entry.input, entry.sub_nid_set.add_input_nid) == 0)
		memcpy(&entry.virtual_position, &layer.GetVirtualPosition(), sizeof(nsas::_VIRTUAL_POSITION));

	_layer_data_info_vector weight_info_vector;
	layer.GetLayerDataInfoVector(weight_info_vector);
	if (ConfigureWeights(weight_info_vector.size(), entry.sub_nid_set.data_nids))
	{
		if (m_reader!=NULL)	// save as일때 기존 weight 값들을 복사
			CopyWeights(layer.GetStoredNidSet().data_nids, entry.sub_nid_set.data_nids);
	}

	// 나머지 부족한 것은 실제 engine에서 읽어 들일때 채우도록 하자.
	if (!m_writer.WriteHiddenLayer(entry))
	{
		DEBUG_OUTPUT(L"failed layer entry");
		return false;
	}

	if(new_nid_set)
		*new_nid_set = entry.sub_nid_set;
	return true;
}

neuro_u32 HiddenLayerWriter::ConfigureInputEntries(const network::_slice_input_vector& input_vector, _INPUT_ENTRY& first_input, neuro_u32& add_input_nid)
{
	if (input_vector.size() > 0)
	{
		first_input.uid = input_vector[0].layer->uid;
		first_input.slice_info = input_vector[0].slice_info;
	}
	else
	{
		first_input.uid = neuro_last32;
	}

	if (input_vector.size() > 1)
	{
		if (add_input_nid == neuro_last32)	// 할당되어 있지 않으면 할당 해야 한다.
		{
			if (!m_writer.AllocDataNda(1, &add_input_nid) || add_input_nid == neuro_last32)
			{
				DEBUG_OUTPUT(L"failed alloc n-node for input");
				return neuro_last32;
			}
		}

		NeuroDataNodeAccessor ndna(m_writer.GetNSAS(), add_input_nid, false);
		NeuroDataAccessManager* nda = ndna.GetWriteAccessor();
		if(nda==NULL)
		{
			DEBUG_OUTPUT(L"no add input nid[%u] accessor", add_input_nid);
			return neuro_last32;
		}

		for (neuro_u32 i = 1, n = input_vector.size(); i < n; i++)
		{
			const network::_SLICE_INPUT& input = input_vector[i];

			_INPUT_ENTRY input_entry;
			input_entry.uid = input.layer->uid;
			input_entry.slice_info = input.slice_info;
			if (!nda->WriteData(&input_entry, sizeof(_INPUT_ENTRY)))
			{
				DEBUG_OUTPUT(L"failed write multi input[%u]");
				nda->DeleteFromCurrent();
				return neuro_last32;
			}
		}
		nda->DeleteFromCurrent();
	}
	else
	{
		if (add_input_nid != neuro_last32)
			m_writer.DeallocDataNda(1, &add_input_nid);
		add_input_nid = neuro_last32;
	}
	return input_vector.size();
}

bool HiddenLayerWriter::ConfigureWeights(neuro_u32 layer_data_count, _LAYER_DATA_NIDS& data_nids)
{
	neuro_u32 weight_nid_count = min(_countof(_LAYER_DATA_NIDS::nid_vector), layer_data_count);
	if (weight_nid_count > data_nids.nid_count)
	{
		if (!m_writer.AllocDataNda(weight_nid_count - data_nids.nid_count
			, data_nids.nid_vector + data_nids.nid_count))
		{
			DEBUG_OUTPUT(L"failed layer data nda nid.");
			return false;
		}
	}
	else if (weight_nid_count < data_nids.nid_count)
	{
		m_writer.DeallocDataNda(data_nids.nid_count - weight_nid_count
			, data_nids.nid_vector + weight_nid_count);
	}
	data_nids.nid_count = weight_nid_count;

	return true;
}

bool HiddenLayerWriter::CopyWeights(const _LAYER_DATA_NIDS& stored_nids, _LAYER_DATA_NIDS& data_nids)
{
	neuro_u32 n = min(stored_nids.nid_count, data_nids.nid_count);
	if (n == 0)
		return true;

	if (m_write_buffer.buffer == NULL)
		m_write_buffer.Alloc(1024 * 1024);

	for (neuro_u32 i = 0; i < n; i++)
	{
		nsas::NeuroDataNodeAccessor source(m_reader->GetNSAS(), stored_nids.nid_vector[i], true);
		nsas::NeuroDataNodeAccessor target(m_writer.GetNSAS(), data_nids.nid_vector[i], false);

		const NeuroDataAccessManager* reader = source.GetReadAccessor();
		NeuroDataAccessManager* writer = target.GetWriteAccessor();

		if (reader == NULL || writer == NULL)
		{
			if (reader == NULL)
				DEBUG_OUTPUT(L"no older accessor");
			if (writer == NULL)
				DEBUG_OUTPUT(L"no new accessor");
			return false;
		}

		// 원래는 reader의 weight 크기가 원래 있어야 할 크기와 같지 않으면 넘어가는 등 불필요한 작업을 피하려 했으나
		// weight history도 있는 등 정확한 것을 알수 없기 때문에 그저 reader의 값을 복사하는 것까지만 한다.
		neuro_u32 write_count = reader->GetSize();
		writer->SetSize(write_count * sizeof(neuron_weight));	// 보다 빨리 처리하기 위해서!

		while (write_count>0)
		{
			neuro_u32 read_count = min(write_count, m_write_buffer.count);

			if (!reader->ReadData(m_write_buffer.buffer, read_count * sizeof(neuron_weight)))
			{
				DEBUG_OUTPUT(L"failed read stored_weight_spec. %llu", read_count);
				return false;
			}

			if (!writer->WriteData(m_write_buffer.buffer, read_count * sizeof(neuron_weight)))
			{
				DEBUG_OUTPUT(L"failed weight_nda.WriteData. %llu", read_count);
				return false;
			}
			write_count -= read_count;
		}

	}
	return true;
}
