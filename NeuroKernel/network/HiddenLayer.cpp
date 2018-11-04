#include "stdafx.h"

#include "HiddenLayer.h"

#include "FcLayerConfigure.h"
#include "ConvLayerConfigure.h"
#include "PoolLayerConfigure.h"
#include "DropoutLayerConfigure.h"
#include "RnnLayerConfigure.h"
#include "BnLayerConfigure.h"
#include "ConcatLayerConfigure.h"

#include "OutputLayer.h"

using namespace np;
using namespace np::network;

HiddenLayer* HiddenLayer::CreateInstance(_layer_type type, neuro_u32 uid)
{
	if (type == _layer_type::input)
		return NULL;

	if (type == _layer_type::output)
		return new OutputLayer(uid);

	HiddenLayer* layer = new HiddenLayer(uid);
	if (!layer->SetLayerType(type))
	{
		delete layer;
		return NULL;
	}
	layer->EntryValidation();
	return layer;
}

void HiddenLayer::CheckOutputTensor()
{
	tensor::TensorShape new_ts = MakeOutTensorShape();
	if (new_ts.IsEqual(m_out_ts))
		return;

	m_out_ts = new_ts;	// 내가 변했으면 내 출력으로 있는 layer들도 변했을수 있으므로!
	__super::CheckOutputTensor();
}

FcLayerConfigure fcLayer;
ConvLayerConfigure convLayer;
PoolLayerConfigure poolingLayer;
DropoutLayerConfigure dropoutLayer;
RnnLayerConfigure rnnLayer;
BnLayerConfigure bnLayer;
ConcatLayerConfigure concatLayer;

bool HiddenLayer::SetLayerType(_layer_type type)
{
	HiddenLayerConfigure* new_configure = NULL;
	switch (type)
	{
	case _layer_type::fully_connected:
		new_configure = &fcLayer;
		break;
	case _layer_type::convolutional:
		new_configure = &convLayer;
		break;
	case _layer_type::pooling:
		new_configure = &poolingLayer;
		break;
	case _layer_type::dropout:
		new_configure = &dropoutLayer;
		break;
	case _layer_type::rnn:
		new_configure = &rnnLayer;
		break;
	case _layer_type::batch_norm:
		new_configure = &bnLayer;
		break;
	case _layer_type::concat:
		new_configure = &concatLayer;
		break;
	}
	if (new_configure == NULL)
		return false;

	m_entry_configure = new_configure;
	return true;
}

bool HiddenLayer::ChangeLayerType(_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION* default_entry, _slice_input_vector* erased_input_vector)
{
	if (!AvailableChangeType())
	{
		DEBUG_OUTPUT(L"not available change");
		return false;
	}

	if (!SetLayerType(layer_type))
	{
		DEBUG_OUTPUT(L"failed set layer type : %s -> %s", ToString(GetLayerType()), ToString(layer_type));
		return false;
	}

	if (default_entry)
	{
		memcpy(&m_entry, default_entry, sizeof(nsas::_LAYER_STRUCTURE_UNION));
		EntryValidation();
	}

	if (AvailableInputCount() < m_input_vector.size())
	{
		for (neuro_32 i = AvailableInputCount(); i < m_input_vector.size(); i++)
		{
			if (erased_input_vector)
				erased_input_vector->push_back(m_input_vector[i]);
			m_input_vector[i].layer->ReleaseOutput(this);
		}

		m_input_vector.resize(AvailableInputCount());
	}

	CheckOutputTensor();
	return true;
}

void HiddenLayer::ChangeEntry(const nsas::_LAYER_STRUCTURE_UNION& entry)
{
	memcpy(&m_entry, &entry, sizeof(nsas::_LAYER_STRUCTURE_UNION));
	CheckChangedEntry();
}

void HiddenLayer::CheckChangedEntry()
{
	EntryValidation();
	CheckOutputTensor();
}

HiddenLayer::HiddenLayer(neuro_u32 uid)
: AbstractLayer(uid)
{
	m_entry_configure = NULL;

	memset(&m_entry, 0, sizeof(nsas::_LAYER_STRUCTURE_UNION));

	m_activation_type = network::_activation_type::none;

	memset(&m_weight_info, 0, sizeof(_LAYER_WEIGHT_INFO));
	m_weight_info.init_type = network::_weight_init_type::Xavier;
	m_weight_info.mult_lr = 1;
	memset(&m_bias_info, 0, sizeof(_LAYER_WEIGHT_INFO));
	m_bias_info.init_type = network::_weight_init_type::Constant;
	m_bias_info.mult_lr = 2;

	m_side_input = NULL;

	SetStoredNidSet(NULL);

	memset(&m_virtual_position, 0, sizeof(nsas::_VIRTUAL_POSITION));
}

HiddenLayer::~HiddenLayer()
{
}

_layer_type HiddenLayer::GetLayerType() const
{
	if (m_entry_configure)
		return m_entry_configure->GetLayerType();
	return _layer_type::unknown;
}

void HiddenLayer::EntryValidation()
{
	if (m_entry_configure)
		m_entry_configure->EntryValidation(m_entry);
}

bool HiddenLayer::HasActivation() const
{
	// output을 출력으로 가지고 있으면 따로 activation이 없다.
	if (IsConnectedOutputLayer())
		return false;

	if (m_entry_configure)
		return m_entry_configure->HasActivation();
	return false;
}

bool HiddenLayer::AvailableChangeActivation() const
{
	if (!HasActivation())
		return false;

	if (m_entry_configure)
		return m_entry_configure->AvailableChangeActivation(m_entry);
	return false;
}

_activation_type HiddenLayer::GetActivation() const
{
	if (!HasActivation())
		return _activation_type::none;

	return m_entry_configure->GetActivationType(m_entry, m_activation_type);
}

neuro_u32 HiddenLayer::AvailableInputCount() const
{
	if (m_entry_configure)
		return m_entry_configure->AvailableInputCount();
	return 0;
}

bool HiddenLayer::AvailableSetSideInput(const HiddenLayer* input) const 
{
	if (m_entry_configure)
		return m_entry_configure->AvailableSetSideInput(*this, input);
	return false;
}

tensor::TensorShape HiddenLayer::MakeOutTensorShape() const
{
	if (m_entry_configure)
		return m_entry_configure->MakeOutTensorShape(*this);

	return tensor::TensorShape();
}

bool HiddenLayer::SetOutTensorShape(const tensor::TensorShape& ts)
{
	if (m_entry_configure)
	{
		if (m_entry_configure->SetOutTensorShape(*this, ts))
		{
			CheckOutputTensor();
			return true;
		}
	}
	return false;
}

neuro_u32 HiddenLayer::GetLayerDataInfoVector(_layer_data_info_vector& info_vector) const
{
	if (m_entry_configure)
		return m_entry_configure->GetLayerDataInfoVector(*this, info_vector);
	return 0;
}

bool HiddenLayer::HasWeight() const
{
	if(m_entry_configure)
		return m_entry_configure->HasWeight();
	return false;
}

network::_weight_init_type HiddenLayer::GetWeightInitType(_layer_data_type type) const
{
	if (!HasWeight())
		return network::_weight_init_type::Zero;

	if (type == _layer_data_type::weight)
		return m_weight_info.init_type;
	else
		return m_bias_info.init_type;
}

bool HiddenLayer::AttachStoredInfo(const nsas::_HIDDEN_LAYER_ENTRY& entry, const nsas::_input_entry_vector& input_vector, const _uid_layer_map& layer_map)
{
	network::_layer_data_info_vector weight_info_vector;
	GetLayerDataInfoVector(weight_info_vector);
	if (entry.sub_nid_set.data_nids.nid_count != weight_info_vector.size())
	{
		DEBUG_OUTPUT(L"the weight nid count[%u] is not %u", entry.sub_nid_set.data_nids.nid_count, weight_info_vector.size());
		return false;
	}

	memcpy(&m_entry, &entry.function, sizeof(nsas::_LAYER_STRUCTURE_UNION));
	EntryValidation();

	if (HasActivation())
		SetActivation((network::_activation_type)entry.basic_info.activation);
	else
		SetActivation(network::_activation_type::none);

	m_weight_info.init_type = (_weight_init_type)entry.basic_info.weight_info.init_type;
	m_weight_info.init_scale = entry.basic_info.weight_info.init_scale;
	m_weight_info.mult_lr = entry.basic_info.weight_info.mult_lr;
	m_weight_info.decay = entry.basic_info.weight_info.decay;

	m_bias_info.init_type = (_weight_init_type)entry.basic_info.bias_info.init_type;
	m_bias_info.init_scale = entry.basic_info.bias_info.init_scale;
	m_bias_info.mult_lr = entry.basic_info.bias_info.mult_lr;
	m_bias_info.decay = entry.basic_info.bias_info.decay;

	SetStoredNidSet(&entry.sub_nid_set);
	SetVirtualPosition(entry.virtual_position);

	for (neuro_u32 i = 0, n = min(AvailableInputCount(), input_vector.size()); i < n; i++)
	{
		const nsas::_INPUT_ENTRY& in_entry = input_vector[i];

		_uid_layer_map::const_iterator find_it = layer_map.find(in_entry.uid);
		if (find_it != layer_map.end())
		{
			_SLICE_INPUT input;
			input.layer = find_it->second;
			input.slice_info = in_entry.slice_info;

			m_input_vector.push_back(input);
			find_it->second->RegisterOutput(this);
		}
	}

	if (entry.side_input.uid != neuro_last32)
	{
		_uid_layer_map::const_iterator it_prev_conn = layer_map.find(entry.side_input.uid);
		if (it_prev_conn != layer_map.end() 
			&& AvailableSetSideInput((HiddenLayer*)it_prev_conn->second))
			SetSideInput((HiddenLayer*)it_prev_conn->second);
	}

	CheckOutputTensor();
	return true;
}

void HiddenLayer::SetStoredNidSet(const nsas::_LAYER_DATA_NID_SET* nid_set)
{
	if (nid_set == NULL)
	{
		memset(&m_stored_sub_nid_set, 0, sizeof(nsas::_LAYER_DATA_NID_SET));
		m_stored_sub_nid_set.add_input_nid = neuro_last32;
	}
	else
		memcpy(&m_stored_sub_nid_set, nid_set, sizeof(nsas::_LAYER_DATA_NID_SET));
}

bool HiddenLayer::AvailableConnectHiddenLayer() const
{
	return true;	
	/*	이미 OutputLayer를 출력으로 가지고 있을 경우 또다른 layer를 출력으로 가지게 할수 없게 하려고 했으나
		굳이 그럴필요 있을까 싶어서 뺌
	return !IsConnectedOutputLayer();
	*/
}

bool HiddenLayer::AvailableConnectOutputLayer() const
{
	return !IsConnectedOutputLayer();
	/*	이미 OutputLayer를 출력으로 가지고 있을 경우 또다른 layer를 출력으로 가지게 할수 없게 하려고 했으나
	굳이 그럴필요 있을까 싶어서 뺌
	return m_output_set.size() == 0;
	*/
}

inline bool HiddenLayer::IsConnectedOutputLayer() const
{
	// 이미 OutputLayer를 출력으로 두고 있으면 Activation을 가지지 않는다.
	_hiddenlayer_set::const_iterator it= m_output_set.begin();
	for (; it != m_output_set.end(); it++)
	{
		if ((*it)->GetLayerType() == _layer_type::output)
			return true;
	}
	return false;
}

bool HiddenLayer::InsertInput(AbstractLayer* input_layer, AbstractLayer* insert_prev)
{
	if (FindInputIndex(input_layer) >= 0)
		return false;

	if (AvailableInputCount() == m_input_vector.size())
	{
		if (m_input_vector.size() == 1)
			DelInput(m_input_vector[0].layer);
		else
			return false;
	}

	_SLICE_INPUT in;
	memset(&in, 0, sizeof(_SLICE_INPUT));
	in.layer = input_layer;

	neuro_u32 insert_index = m_input_vector.size();
	if(insert_prev!=NULL)
	{
		for (neuro_u32 i = 0; i < m_input_vector.size(); i++)
		{
			if (m_input_vector[i].layer == insert_prev)
			{
				insert_index = i;
				break;
			}
		}
	}
	m_input_vector.insert(m_input_vector.begin() + insert_index, in);

	input_layer->RegisterOutput(this);

	OnInsertedInput(input_layer);

	CheckOutputTensor();
	return true;
}

bool HiddenLayer::BatchAppendInputs(_slice_input_vector input_vector)
{
	if (AvailableInputCount() < m_input_vector.size() + input_vector.size())
	{
		DEBUG_OUTPUT(L"Exceeding the allowable number of inputs(%u). current[%u] and adding[%u]"
		, AvailableInputCount(), m_input_vector.size(), input_vector.size());
		return false;
	}

	std::unordered_set<AbstractLayer*> input_set;
	for (neuro_u32 i = 0; i < m_input_vector.size(); i++)
		input_set.insert(m_input_vector[i].layer);

	for (neuro_u32 i = 0; i < input_vector.size(); i++)
	{
		if (input_set.find(input_vector[i].layer) != input_set.end())
		{
			DEBUG_OUTPUT(L"input vector has already %th adding layer", i);
			continue;
		}
		m_input_vector.push_back(input_vector[i]);
		input_vector[i].layer->RegisterOutput(this);
	}

	CheckOutputTensor();
	return true;
}

bool HiddenLayer::DelInput(AbstractLayer* input_layer)
{
	if (!ReleaseInput(input_layer))
		return false;

	input_layer->ReleaseOutput(this);
	return true;
}

bool HiddenLayer::ReleaseInput(AbstractLayer* input_layer)
{
	if (m_side_input == input_layer)
	{
		m_side_input = NULL;
	}
	else
	{
		int index = FindInputIndex(input_layer);
		if (index<0)
			return false;

		m_input_vector.erase(m_input_vector.begin() + index);
	}

	CheckOutputTensor();
	return true;
}

int HiddenLayer::FindInputIndex(AbstractLayer* input_layer) const
{
	for (int i = 0; i < m_input_vector.size(); i++)
	{
		if (input_layer == m_input_vector[i].layer)
			return i;
	}
	return -1;
}

bool HiddenLayer::SetSideInput(HiddenLayer* input)
{
	if (input != NULL && !AvailableSetSideInput(input))
		return false;

	m_side_input = input;
	m_side_input->RegisterOutput(this);
	CheckOutputTensor();
	return true;
}

void HiddenLayer::OnRemove()
{
	// 모든 입력 layer들에 대해 출력중에 나를 제거하라고 해줘야 한다.
	for (_slice_input_vector::iterator it = m_input_vector.begin(), end = m_input_vector.end(); it != end; it++)
		it->layer->ReleaseOutput(this);

	__super::OnRemove();
}

