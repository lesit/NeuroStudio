#include "stdafx.h"

#include "NeuralNetwork.h"

#include "util/np_util.h"

#include "loader/NeuralNetworkLoader.h"
#include "writer/NeuralNetworkWriter.h"

using namespace np::network;

NeuralNetwork::NeuralNetwork()
{
	m_load_device=NULL;	// 로드한 device. 없으면 새로 저장하기이다.
	m_load_nsas=NULL;

	memset(&m_input_layers, 0, sizeof(_LINKED_LAYER));
	memset(&m_hidden_layers, 0, sizeof(_LINKED_LAYER));

	memset(&m_learning_info, 0, sizeof(_LEARNING_INFO));
	m_learning_info.optimizer_type = _optimizer_type::SGD;
}

NeuralNetwork::~NeuralNetwork()
{
	ClearAll();
}

void NeuralNetwork::ClearAll()
{
	AbstractLayer* layer = m_input_layers.start;
	while (layer)
	{
		AbstractLayer* next = layer->GetNext();
		delete layer;
		layer = next;
	}
	layer = m_hidden_layers.start;
	while (layer)
	{
		AbstractLayer* next = layer->GetNext();
		delete layer;
		layer = next;
	}

	m_id_factory.RemoveAll();

	memset(&m_input_layers, 0, sizeof(_LINKED_LAYER));
	memset(&m_hidden_layers, 0, sizeof(_LINKED_LAYER));
	m_layer_map.clear();
	m_input_layer_set.clear();
	m_output_layer_set.clear();

	m_deleted_layer_data_nid_vector.clear();

	if (m_load_nsas)
		delete m_load_nsas;
	m_load_nsas = NULL;

	if (m_load_device)
		delete m_load_device;
	m_load_device = NULL;
}

void NeuralNetwork::New()
{
	ClearAll();

	memset(&m_learning_info, 0, sizeof(_LEARNING_INFO));
	m_learning_info.optimizer_type = _optimizer_type::SGD;

//	AddHiddenLayer();	// output layer
}

bool NeuralNetwork::Load(device::IODeviceFactory& device)
{
	DEBUG_OUTPUT(L"start");

	device::DeviceAdaptor* load_device = device.CreateWriteAdaptor(false, false, 0);
	if (!load_device)
	{
		DEBUG_OUTPUT(L"failed create device");
		delete load_device;
		return false;
	}

	nsas::NeuroStorageAllocationSystem* load_nsas=new nsas::NeuroStorageAllocationSystem(*load_device);
	if (!load_nsas->LoadNSAS())
	{
		DEBUG_OUTPUT(L"failed LoadNSAS");
		delete load_device;
		delete load_nsas;
		return false;
	}

	ClearAll();

	m_load_device = load_device;
	m_load_nsas = load_nsas;

	loader::NeuralNetworkLoader loader(*m_load_nsas, m_id_factory, m_layer_map, m_input_layer_set, m_output_layer_set);
	if (!loader.Load(m_learning_info, m_input_layers, m_hidden_layers))
	{
		DEBUG_OUTPUT(L"failed LoadNetwork");
		New();
		return false;
	}

	if (!LoadCompleted())
	{
		DEBUG_OUTPUT(L"failed LoadCompleted");
		New();
		return false;
	}
	DEBUG_OUTPUT(L"end");
	return true;
}

// OnNeuralNetworkReplace 에서만 bReload=false
// 일반 저장은 이전에 로드된것이 있어야만 가능하다. 새로 저장은 SaveAs로만 가능하다.
bool NeuralNetwork::Save(bool apply_nsas_to_layer)
{
	DEBUG_OUTPUT(L"start");

	if (!m_load_nsas)
	{
		DEBUG_OUTPUT(L"no load nsas");
		return false;
	}

	writer::NeuralNetworkWriter writer(*this);
	if (!writer.Save(m_load_nsas, *m_load_nsas, apply_nsas_to_layer))
	{
		DEBUG_OUTPUT(L"failed to WriteNetwork");
		// 실패할 경우 nsas를 초기화하고 각 hidden layer의 stored nid set이 없는 상태로 다시 저장해야 한다.
		m_load_nsas->InitNSAS(m_load_nsas->GetBlockSize());
		if (!writer.Save(NULL, *m_load_nsas, apply_nsas_to_layer))
		{
			DEBUG_OUTPUT(L"again failed to WriteNetwork");

		}
		return false;
	}

	DEBUG_OUTPUT(L"end");
	return true;
}

bool NeuralNetwork::SaveAs(device::IODeviceFactory& device, neuro_u32 block_size, bool apply_nsas_to_layer)
{
	DEBUG_OUTPUT(L"start. block size %u", block_size);

	device.Reset();
	device::DeviceAdaptor* save_device = device.CreateWriteAdaptor(true, false);
	if (!save_device)
	{
		DEBUG_OUTPUT(L"failed create write adaptor");
		return false;
	}

	NeuroStorageAllocationSystem* save_nsas = new NeuroStorageAllocationSystem(*save_device);
	if (!save_nsas->InitNSAS(block_size))
	{
		DEBUG_OUTPUT(L"Failed to CreateNAS\r\n");

		delete save_device;
		delete save_nsas;
		return false;
	}

	writer::NeuralNetworkWriter writer(*this);
	if (!writer.Save(m_load_nsas, *save_nsas, apply_nsas_to_layer))
	{
		DEBUG_OUTPUT(L"Failed to WriteNetwork\r\n");

		delete save_device;
		delete save_nsas;
		return false;
	}
	m_deleted_layer_data_nid_vector.clear();

	delete m_load_device;
	delete m_load_nsas;

	m_load_device = save_device;
	m_load_nsas = save_nsas;

	DEBUG_OUTPUT(L"end");
	return true;
}

AbstractLayer* NeuralNetwork::CreateLayerInstance(network::_layer_type type)
{
	neuro_u32 uid = m_id_factory.CreateId();
	if (uid == neuro_last32)
		return NULL;

	AbstractLayer* layer;
	if (type == network::_layer_type::input)
		layer = new InputLayer(uid, tensor::TensorShape());
	else
		layer = HiddenLayer::CreateInstance(type, uid);
	if (layer == NULL)
	{
		m_id_factory.RemoveId(uid);
		return NULL;
	}
	return layer;
}

void NeuralNetwork::DestroyLayerInstance(AbstractLayer* layer)
{
	m_id_factory.RemoveId(layer->uid);
	delete layer;
}

bool NeuralNetwork::AddLayerInLinkedList(AbstractLayer* layer, AbstractLayer* insert_prev)
{
	if (insert_prev)
	{
		if ((layer->GetLayerType() == _layer_type::input) != (insert_prev->GetLayerType() == _layer_type::input))
			return false;
	}

	network::_layer_type type = layer->GetLayerType();

	m_layer_map[layer->uid] = layer;
	if (type == network::_layer_type::input)
		m_input_layer_set.insert((InputLayer*)layer);
	if (type == network::_layer_type::output)
		m_output_layer_set.insert((OutputLayer*)layer);

	_LINKED_LAYER& linked_layer = type == network::_layer_type::input ? m_input_layers : m_hidden_layers;

	AbstractLayer *prev = NULL;
	AbstractLayer *next = NULL;
	if(insert_prev)
	{
		prev = insert_prev->GetPrev();
		next = insert_prev;
	}
	else
		prev = linked_layer.end;

	if (prev)
		prev->SetNext(layer);
	else
		linked_layer.start = layer;
	if (next)
		next->SetPrev(layer);
	else
		linked_layer.end = layer;

	++linked_layer.count;

	layer->SetPrev(prev);
	layer->SetNext(next);

	return true;
}

void NeuralNetwork::RemoveLayerInLinkedList(AbstractLayer* layer)
{
	if (layer == NULL)
		return;

	m_layer_map.erase(layer->uid);

	network::_layer_type type = layer->GetLayerType();
	if (type == network::_layer_type::input)
		m_input_layer_set.erase((InputLayer*)layer);
	if (layer->GetLayerType() == network::_layer_type::output)
		m_output_layer_set.erase((OutputLayer*)layer);

	_LINKED_LAYER& linked_layer = type == network::_layer_type::input ? m_input_layers : m_hidden_layers;

	AbstractLayer *prev = layer->GetPrev();
	AbstractLayer *next = layer->GetNext();
	if (prev)
		prev->SetNext(next);
	if (next)
		next->SetPrev(prev);

	if (layer == linked_layer.start)
		linked_layer.start = next;
	if (layer == linked_layer.end)
		linked_layer.end = prev;

	--linked_layer.count;
}

AbstractLayer* NeuralNetwork::AddLayer(np::network::_layer_type type, AbstractLayer* insert_prev)
{
	if (insert_prev)
	{
		if ((type == _layer_type::input) != (insert_prev->GetLayerType() == _layer_type::input))
			return NULL;
	}

	AbstractLayer* layer = CreateLayerInstance(type);
	if (!layer)
		return NULL;
	AddLayerInLinkedList(layer, insert_prev);
	return layer;
}

bool NeuralNetwork::DeleteLayer(AbstractLayer* layer)
{
	layer->OnRemove();

	if(layer->GetLayerType() != network::_layer_type::input)
		m_deleted_layer_data_nid_vector.push_back(((const HiddenLayer*)layer)->GetStoredNidSet());

	RemoveLayerInLinkedList(layer);
	DestroyLayerInstance(layer);

	return true;
}

bool NeuralNetwork::MoveLayerTo(AbstractLayer* layer, AbstractLayer* insert_prev)
{
	if (layer==NULL || layer == insert_prev)
		return false;

	RemoveLayerInLinkedList(layer);
	AddLayerInLinkedList(layer, insert_prev);
	return true;
}

bool NeuralNetwork::ConnectTest(AbstractLayer* from_layer, AbstractLayer* to_layer)
{
	if (to_layer->GetLayerType() == network::_layer_type::input)
		return false;

	if (to_layer->GetLayerType() == network::_layer_type::output)
	{
		// 출력의 layer가 OutputLayer일 때 입력의 layer와 연결될수 있는지.
		if (!from_layer->AvailableConnectOutputLayer())
			return false;
	}
	else if (from_layer->GetLayerType() != network::_layer_type::input)
	{
		// 입력이 hidden layer일 때 hidden layer와 연결 될 수 있는지
		if (!((HiddenLayer*)from_layer)->AvailableConnectHiddenLayer())
			return false;
	}
	return ((HiddenLayer*)to_layer)->FindInputIndex(from_layer)<0;
}

bool NeuralNetwork::Connect(AbstractLayer* from_layer, AbstractLayer* to_layer, AbstractLayer* insert_prev)
{
	if (to_layer->GetLayerType() == network::_layer_type::input)
		return false;

	return ((HiddenLayer*)to_layer)->InsertInput(from_layer, insert_prev);
}

bool NeuralNetwork::SideConnectTest(AbstractLayer* from_layer, AbstractLayer* to_layer)
{
	if (from_layer->GetLayerType() == network::_layer_type::input)
		return false;

	return ((HiddenLayer*)to_layer)->AvailableSetSideInput((HiddenLayer*)from_layer);
}

bool NeuralNetwork::SideConnect(AbstractLayer* from_layer, AbstractLayer* to_layer)
{
	if (!SideConnectTest(from_layer, to_layer))
		return false;

	return ((HiddenLayer*)to_layer)->SetSideInput((HiddenLayer*)from_layer);
}

bool NeuralNetwork::DisConnect(AbstractLayer* from_layer, AbstractLayer* to_layer)
{
	if (to_layer->GetLayerType() == network::_layer_type::input)
		return false;

	return ((HiddenLayer*)to_layer)->DelInput(from_layer);
}

bool NeuralNetwork::ConnectTest(NetworkBindingModel* binding, AbstractLayer* to_layer)
{
	if (binding == NULL || to_layer == NULL || to_layer->GetLayerType()!=_layer_type::input && to_layer->GetLayerType()!=_layer_type::output)
		return false;

	if (binding->GetBindingSet().size() > 0)
	{
		// 이미 있으면 안된다.
		return false;
	}

	const _neuro_binding_model_set& binding_set = to_layer->GetBindingSet();
	if (binding_set.find(binding) != binding_set.end())
		return false;

	if (binding->GetBindingModelType() == _binding_model_type::data_producer)
	{
		dp::model::AbstractProducerModel* producer = (dp::model::AbstractProducerModel*) binding;

		const dp::model::DataProviderModel& provider = producer->GetProvider();

		if (to_layer->GetLayerType() == network::_layer_type::input)
		{
			if (!producer->AvailableToInputLayer())
				return false;

			/*	InputLayer는 두개의 binding을 가질 수 있지만
			다른 타입(predict, learn)의 provider에 있는 것을 각각 한개씩 가질 수 있다.
			*/
			_neuro_binding_model_set::const_iterator it = binding_set.begin();
			for (; it != binding_set.end(); it++)
			{
				NetworkBindingModel* other = *it;
				if (other->GetBindingModelType() == _binding_model_type::data_producer)
				{
					const dp::model::DataProviderModel& other_provider = ((dp::model::AbstractProducerModel*) other)->GetProvider();
					if (&provider == &other_provider)
						return false;
				}
			}
			
		}
		else if (to_layer->GetLayerType() == network::_layer_type::output)
		{
			// learn producer만 연결 할 수 있다.
			if (!provider.IsLearnProvider())
				return false;

			if (!producer->AvailableToOutputLayer())
				return false;

			// OutputLayer는 하나의 producer binding만 가질 수 있다.
			_neuro_binding_model_set::const_iterator it = binding_set.begin();
			for (; it != binding_set.end(); it++)
			{
				NetworkBindingModel* other = *it;
				if (other->GetBindingModelType() == _binding_model_type::data_producer)
					return false;
			}
		}
		else
			return false;
	}

	return true;
}

bool NeuralNetwork::Connect(NetworkBindingModel* binding, AbstractLayer* to_layer)
{
	if (!ConnectTest(binding, to_layer))
		return false;

	((NetworkBindingModel*)to_layer)->AddBinding(binding);
	return true;
}

bool NeuralNetwork::DisConnect(NetworkBindingModel* binding, AbstractLayer* to_layer)
{
	if (to_layer->GetLayerType() != _layer_type::input && to_layer->GetLayerType() != _layer_type::output)
		return false;

	((NetworkBindingModel*)to_layer)->RemoveBinding(binding);
	return true;
}
