#include "stdafx.h"

#include "NeuralNetworkEngine.h"

#include "util/np_util.h"

#include "WeightStoreManager.h"
#include "SharedDataBuffers.h"

#include "core/cuda_platform.h"

using namespace np::engine;

_PARALLEL_INSTANCE::_PARALLEL_INSTANCE()
{
	DEBUG_OUTPUT(L"start");

	cuda_instance = core::cuda::CudaPlatform::CreateInstance();
	if (cuda_instance == NULL)
	{
		DEBUG_OUTPUT(L"no cuda instance");
	}

	DEBUG_OUTPUT(L"end");
}

_PARALLEL_INSTANCE::~_PARALLEL_INSTANCE()
{
	if (cuda_instance)
		core::cuda::CudaPlatform::DestoryInstance(cuda_instance);
}

bool _PARALLEL_INSTANCE::IsAvailable() const
{
	return cuda_instance!=NULL;
}

NeuralNetworkEngine* NeuralNetworkEngine::CreateInstance(core::math_device_type pdType, _PARALLEL_INSTANCE& p_instance, network::NeuralNetwork& network)
{
	if (network.GetLoadNSAS() == NULL)
	{
		DEBUG_OUTPUT(L"no loaded nsas");
		return NULL;
	}

	WeightStoreManager* wsm = new WeightStoreManager(pdType, *network.GetLoadNSAS());
	if (wsm == NULL)
	{
		DEBUG_OUTPUT(L"failed create weight store manager");
		return NULL;
	}

	NeuralNetworkEngine* engine = new NeuralNetworkEngine(pdType, p_instance, network.GetLearningInfo(), wsm);
	if (!engine->Load(network))
	{
		delete engine;
		return NULL;
	}

	return engine;
}

NeuralNetworkEngine::NeuralNetworkEngine(core::math_device_type pdType, _PARALLEL_INSTANCE& p_instance
	, const network::_LEARNING_INFO& learning_info, WeightStoreManager* wsm)
: m_net_param(pdType, p_instance.cuda_instance)
, m_learning_info(learning_info)
{
	m_weightStoreManager = wsm;

	m_load_nsas = NULL;

	m_total_layer_inner_data_size = 0;
	m_total_layer_out_size = 0;

	memset(&m_buffer_alloc_info, 0, sizeof(_BUFFER_ALLOC_INFO));
}

NeuralNetworkEngine::~NeuralNetworkEngine()
{
	delete m_weightStoreManager;

	for (_input_engine_vector::iterator it = m_input_engine_vector.begin(), end = m_input_engine_vector.end(); it != end; it++)
		delete *it;
	for (_hidden_engine_vector::iterator it = m_hidden_engine_vector.begin(), end = m_hidden_engine_vector.end(); it != end; it++)
		delete *it;
}

#include "../network/HiddenLayer.h"
#include "layers/BatchNormLayerEngine.h"
#include "layers/ConcatLayerEngine.h"
#include "layers/ConvLayerEngine.h"
#include "layers/DropoutLayerEngine.h"
#include "layers/FcLayerEngine.h"
#include "layers/PoolingLayerEngine.h"
#include "layers/RecurrentLayerEngine.h"
#include "layers/OutputLayerEngine.h"

bool NeuralNetworkEngine::Load(network::NeuralNetwork& network)
{
	_engine_map engine_map;

	const network::_LINKED_LAYER& input_layers = network.GetInputLayers();
	network::AbstractLayer* layer = input_layers.start;
	while (layer)
	{
		InputLayerEngine* engine = new InputLayerEngine(m_net_param, (network::InputLayer&)*layer);
		m_input_engine_vector.push_back(engine);

		m_total_layer_out_size += engine->GetOutTensorShape().GetTensorSize();

		engine_map[layer] = engine;
		m_uid_engine_map[layer->uid] = engine;

		layer = layer->GetNext();
	}
	layer = network.GetHiddenLayers().start;
	while (layer)
	{
		HiddenLayerEngine* engine = CreateHiddenLayerEngine(*(network::HiddenLayer*)layer, engine_map);
		if (engine == NULL)
		{
			DEBUG_OUTPUT(L"failed create hidden layer engine");
			return false;
		}

		m_net_param.sdb.SetLayerOnesetSize(engine->Get1MultiplierSize(), engine->Get1MultiplierSizePerBatch());
		
		const _layer_data_vector& layer_data_vector = engine->GetInnerDataVector();
		for (neuro_u32 i = 0; i < layer_data_vector.size(); i++)
		{
			const _LAYER_INNER_DATA& layer_data = layer_data_vector[i];
			m_total_layer_inner_data_size += layer_data.data.count + layer_data.snapshot.count;
		}
		m_total_layer_out_size += engine->GetOutTensorShape().GetTensorSize();

		m_hidden_engine_vector.push_back(engine);
		if (engine->GetLayerType()==_layer_type::output)
			m_output_engine_vector.push_back((OutputLayerEngine*)engine);

		engine_map[layer] = engine;
		m_uid_engine_map[layer->uid] = engine;

		layer = layer->GetNext();
	}

	if (!m_weightStoreManager->ReadAllWeights())
	{
		DEBUG_OUTPUT(L"failed load weights");
		return false;
	}

	m_load_nsas = network.GetLoadNSAS();
	return true;
}

HiddenLayerEngine* NeuralNetworkEngine::CreateHiddenLayerEngine(const network::HiddenLayer& layer, const _engine_map& engine_map)
{
	{
		network::_layer_data_info_vector layer_data_info_vector;
		layer.GetLayerDataInfoVector(layer_data_info_vector);
		if (layer_data_info_vector.size() != layer.GetStoredNidSet().data_nids.nid_count)
		{
			DEBUG_OUTPUT(L"weight size vector is not equal with nid set");
			return false;
		}
	}

	const network::_slice_input_vector& layer_input_vector = layer.GetInputVector();

	_input_vector input_vector;
	input_vector.resize(layer_input_vector.size());

	for (neuro_u32 i = 0; i < layer_input_vector.size(); i++)
	{
		const network::_SLICE_INPUT& layer_input = layer_input_vector[i];
		_engine_map::const_iterator it = engine_map.find(layer_input.layer);
		if (it == engine_map.end())
		{
			DEBUG_OUTPUT(L"not found input engine");
			return NULL;
		}

		layers::_INPUT_INFO& input_info = input_vector[i];
		input_info.engine = it->second;
		input_info.slice_info = layer_input.slice_info;
	}

	HiddenLayerEngine* side_layer_engine = NULL;
	if(layer.GetSideInput())
	{
		_engine_map::const_iterator it = engine_map.find(layer.GetSideInput());
		if (it != engine_map.end())
			side_layer_engine = (HiddenLayerEngine*)it->second;
	}

	HiddenLayerEngine* engine = NULL;

	switch (layer.GetLayerType())
	{
	case network::_layer_type::fully_connected:
		engine = FcLayerEngine::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::convolutional:
		engine = ConvLayerEngineBase::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::pooling:
		engine = PoolingLayerEngine::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::dropout:
		engine = DropoutLayerEngine::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::rnn:
		engine = RecurrentLayerEngine::CreateInstance(m_net_param, layer, side_layer_engine);
		break;
	case network::_layer_type::batch_norm:
		engine = BatchNormLayerEngine::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::concat:
		engine = ConcatLayerEngine::CreateInstance(m_net_param, layer);
		break;
	case network::_layer_type::output:
		engine = OutputLayerEngine::CreateInstance(m_net_param, layer);
		break;
	}

	if (!engine)
	{
		DEBUG_OUTPUT(L"invalid engine type");
		return NULL;
	}

	if (!engine->Initialize(input_vector))
	{
		DEBUG_OUTPUT(L"failed to Initialize");
		delete engine;
		return NULL;
	}

	if (engine->GetInnerDataVector().size()>0)
		m_weightStoreManager->RegisterLayerWeight(engine);

	return engine;
}

bool NeuralNetworkEngine::GetLayerOutInfo(neuro_u64 uid, _LAYER_SCALE_INFO& info) const
{
	memset(&info, 0, sizeof(_LAYER_SCALE_INFO));

	_uid_engine_map::const_iterator it = m_uid_engine_map.find(uid);
	if (it == m_uid_engine_map.end())
		return false;

	const layers::AbstractLayerEngine* layer = it->second;

	std::pair<neuron_value, neuron_value> scale = layer->GetOutputScale();
	info.low_scale = scale.first;
	info.up_scale = scale.second;

	return true;
}

bool NeuralNetworkEngine::AllocBuffers(neuro_u32 batch_size)
{
	DEBUG_OUTPUT(L"start");
	if (batch_size == 0)
	{
		DEBUG_OUTPUT(L"no batch");
		return false;
	}

	if (m_buffer_alloc_info.batch_size == batch_size)
	{
		DEBUG_OUTPUT(L"already allocated");
		return true;
	}

	DEBUG_OUTPUT(L"alloc shared buffers");
	if (!m_net_param.sdb.InitializeBuffer(batch_size))
	{
		DEBUG_OUTPUT(L"failed to InitializeBuffer of shared buffer");
		return false;
	}

	memset(&m_buffer_alloc_info, 0, sizeof(_BUFFER_ALLOC_INFO));

	DEBUG_OUTPUT(L"dealloc layer buffers");
	for (neuro_u32 i = 0; i < m_input_engine_vector.size(); i++)
		m_input_engine_vector[i]->DeallocOutputBuffer();

	for (neuro_u32 i = 0; i < m_hidden_engine_vector.size(); i++)
		m_hidden_engine_vector[i]->DeallocOutputBuffer();

	DEBUG_OUTPUT(L"dealloc buffers end");

	// 이때 gpu의 memory 할당 방식을 결정하자. 즉, gpu full memory인지 아님 필요할때만 할당해서
	// cpu memory로부터 필요할때만 읽어 들일지!
	// 만약 gpu full memory인 경우 모든 layer에는 gpu memory만 있으면 된다.
	// 만약 부분 goup memory인 경우 모든 layer에는 cpu memory만 할당되어 있고,
	// 각 layer에 대한 계산을 할때 cpu memory로 부터 gpu memory로 해당 부분(입력 데이터)를 읽어 들인다.
	// 즉, 이땐 gpu memory는 공유해서 사용하도록 한다.
	// 일단 무조건 gpu full memory로 하고 나중에 생각해보자!

	core::math_device_type layer_outbuf_pd_type = m_net_param.run_pdtype;
	if (m_net_param.run_pdtype == core::math_device_type::cuda)
	{
		if (!m_net_param.cuda_instance)
		{
			DEBUG_OUTPUT(L"no cuda platform");
			return false;
		}

		neuro_size_t free, total;
		if (!core::cuda::CudaPlatform::GetMemoryInfo(free, total))
		{
			DEBUG_OUTPUT(L"failed get memory info from cuda. error:%s", core::cuda::CudaPlatform::GetErrorString());
			return false;
		}

		if (free < m_total_layer_out_size*batch_size*1.5)
		{
			layer_outbuf_pd_type = core::math_device_type::cpu;
			DEBUG_OUTPUT(L"changed from cuda to cpu mode. memory of gpu is not enough for all layer's output."
				L" gpu free[%llu] must be over than total layer output[%llu] X batch[%llu] X 1.5"
				, free, m_total_layer_out_size, batch_size);
		}
		/*
#ifdef _DEBUG
		// m_shared_data_buffer->InitializeBuffer에서 의외로 많은 메모리를 사용한다.
		// 정확한 계산을 할수 있을때까지 이렇게 하자!ㅠㅠ
		layer_outbuf_pd_type = core::math_device_type::cpu;
#endif
		*/
	}

	DEBUG_OUTPUT(L"alloc layer buffers");
	for (neuro_u32 i = 0; i < m_input_engine_vector.size(); i++)
	{
		if (!m_input_engine_vector[i]->AllocOutputBuffer(layer_outbuf_pd_type, batch_size))
		{
			DEBUG_OUTPUT(L"failed alloc buffer. %uth input layer", i);
			return false;
		}
	}
	for (neuro_u32 i = 0; i < m_hidden_engine_vector.size(); i++)
	{
		if (!m_hidden_engine_vector[i]->AllocOutputBuffer(layer_outbuf_pd_type, batch_size))
		{
			DEBUG_OUTPUT(L"failed alloc buffer. %uth hidden layer", i);
			return false;
		}
	}

	m_buffer_alloc_info.batch_size = batch_size;
	return true;
}
