#include "stdafx.h"

#include "HiddenLayerEngine.h"

#include "util/StringUtil.h"

#include "util/cpu_parallel_for.h"
#include "../backend/loss_function.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;

HiddenLayerEngine::HiddenLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: AbstractLayerEngine(net_param, layer)
, m_error_buffer(net_param.run_pdtype, true)
, m_entry(layer.GetEntry())
{
	m_activation = NULL;
}

HiddenLayerEngine::~HiddenLayerEngine()
{
	if (m_activation)
		delete m_activation;

	for (neuro_u32 i = 0; i < m_inner_data_vector.size(); i++)
		m_inner_data_vector[i].Dealloc();
}

bool HiddenLayerEngine::Initialize(const _input_vector& input_vector)
{
	if (MustHaveInput() && input_vector.size() == 0)
	{
		DEBUG_OUTPUT(L"not input");
		return false;
	}

	m_input_vector = input_vector;
	m_in_ts = m_input_vector[0].slice_info.GetTensor(m_input_vector[0].engine->GetOutTensorShape());

	const nsas::_LAYER_DATA_NID_SET& layer_data_nid_set = ((network::HiddenLayer&)m_layer).GetStoredNidSet();

	network::_layer_data_info_vector layer_data_info_vector;
	((HiddenLayer&)m_layer).GetLayerDataInfoVector(layer_data_info_vector);

	if (layer_data_nid_set.data_nids.nid_count != layer_data_info_vector.size() ||
		layer_data_nid_set.data_nids.nid_count > _countof(layer_data_nid_set.data_nids.nid_vector))
	{
		DEBUG_OUTPUT(L"weight nid count is not equal with weight init vector's");
		return false;
	}

	m_inner_data_vector.resize(layer_data_info_vector.size());
	for (neuro_u32 i = 0; i < layer_data_info_vector.size(); i++)
	{
		_LAYER_INNER_DATA& innder_data = m_inner_data_vector[i];
		innder_data.wtype = layer_data_info_vector[i].wtype;
		innder_data.nid = layer_data_nid_set.data_nids.nid_vector[i];
		if(!innder_data.Alloc(m_net_param.run_pdtype, layer_data_info_vector[i].size))
		{
			DEBUG_OUTPUT(L"failed alloc weight buffer");
			return false;
		}
	}

	_activation_type activation_type = ((HiddenLayer&)m_layer).GetActivation();
	if (activation_type != network::_activation_type::none)
	{
		if (m_net_param.run_pdtype == core::math_device_type::cuda)
			m_activation = activation::ActivationFunction::CreateInstanceCUDA(activation_type);
		else
			m_activation = activation::ActivationFunction::CreateInstanceCPU(activation_type);
		if (m_activation == NULL)
		{
			DEBUG_OUTPUT(L"failed create activation instance.");
			return false;
		}
	}

	return OnInitialized();
}

std::pair<neuron_value, neuron_value> HiddenLayerEngine::GetOutputScale() const
{
	if (m_activation)
		return std::make_pair(m_activation->GetScaleMin(), m_activation->GetScaleMax());

	if (m_input_vector.size() > 0)
		return m_input_vector[0].engine->GetOutputScale();

	return __super::GetOutputScale();
}

bool HiddenLayerEngine::GetInputData(const _INPUT_INFO& input, _NEURO_TENSOR_DATA& buffer) const
{
	if(!m_net_param.GetDeviceBuffer(input.engine->GetOutputData(), TensorBatchTimeOrder(), buffer))
		return false;

	// input의 slice에 따른 데이터를 복사해줘야 한다.
	return true;
}

bool HiddenLayerEngine::Propagation(bool bTrain, neuro_u32 batch_size)
{
	const _NEURO_TENSOR_DATA& output_buffer = GetOutputData();

	if (output_buffer.data.mm.GetType() == m_net_param.run_pdtype)
		return Forward(bTrain, batch_size, output_buffer);

	_NEURO_TENSOR_DATA temp(m_net_param.run_pdtype, true);
	temp.AllocLike(output_buffer);
	temp.SetZero();

	if(!Forward(bTrain, batch_size, temp))
		return false;

	output_buffer.CopyFrom(temp);
	return true;
}

// 이제 weight buffer는 각 layer에서 알아서 사용하도록 한다!
bool HiddenLayerEngine::Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output_data)
{
	_NEURO_TENSOR_DATA input;
	if (!GetInputData(m_input_vector[0], input))
	{
		DEBUG_OUTPUT(L"no input data");
		return false;
	}

	if (!ForwardData(bTrain, input, output_data))
	{
		DEBUG_OUTPUT(L"%u, %u. failed layer forward.");
		return false;
	}

	if (m_activation)
	{
		if (!m_activation->ForwardActivations(output_data))
		{
			DEBUG_OUTPUT(L"%s, failed ForwardActivations : %s", GetLayerName(), core::cuda::CudaPlatform::GetErrorString().c_str());
			return false;
		}
	}
	return true;
}

bool HiddenLayerEngine::Backpropagation(neuro_u32 batch_size)
{
	bool ret = Backward(batch_size);

	m_error_buffer.Dealloc();	// error 에 대한 계산을 다 했으면 초기화해서 메모리 할당량을 줄여주자!
	return ret;
}

bool HiddenLayerEngine::Backward(neuro_u32 batch_size)
{
#if defined(_DEBUGOUT_LAYER_BACKWARD)
	DEBUG_OUTPUT(L"%s : %u, %u", GetLayerName(), uid.upper_uid(), uid);
	if (GetLayerType() == network::_layer_type::pooling)
		int a = 0;
#endif

	_NEURO_TENSOR_DATA output_data;
	if (!m_net_param.GetDeviceBuffer(GetOutputData(), TensorBatchTimeOrder(), output_data))
	{
		DEBUG_OUTPUT(L"failed to get data buffer");
		return false;
	}

	if (m_error_buffer.GetSize() != output_data.GetSize())
	{
		DEBUG_OUTPUT(L"error buffer[%llu] is not allocated like output buffer[%llu]."
		, m_error_buffer.GetSize(), output_data.GetSize());
		return false;
	}

	// recurrent와 같이 data order가 바뀌는 경우를 대비함
	_NEURO_TENSOR_DATA error_data;
	if (!m_net_param.GetDeviceBuffer(m_error_buffer, TensorBatchTimeOrder(), error_data))
	{
		DEBUG_OUTPUT(L"failed to get data buffer");
		return false;
	}

	if (m_activation)
	{
		if (!m_activation->BackwardActivations(output_data, error_data))
		{
			DEBUG_OUTPUT(L"failed Backward");
			return false;
		}
	}

	_NEURO_TENSOR_DATA input_data;
	if (!GetInputData(m_input_vector[0], input_data))
	{
		DEBUG_OUTPUT(L"no input data");
		return false;
	}

	if (m_input_vector[0].engine->GetLayerType() != network::_layer_type::input)
	{
		const _NEURO_TENSOR_DATA& input_error = ((HiddenLayerEngine*)m_input_vector[0].engine)->GetErrorBuffer();

		// output_data는 PoolingLayerCudaEngine 에서만 사용한다!!
		if (!BackwardError(output_data, error_data, input_data, input_error))
		{
			DEBUG_OUTPUT(L"failed BackwardError");
			return false;
		}
	}

	if (((network::HiddenLayer&)m_layer).HasWeight())
	{
		_VALUE_VECTOR grad_weight(m_net_param.run_pdtype, true);
		for (neuro_u32 i = 0; i < m_inner_data_vector.size(); i++)
		{
			const _LAYER_INNER_DATA& inner_data = m_inner_data_vector[i];
			const _LAYER_WEIGHT_INFO* info = ((HiddenLayer&)m_layer).GetWeightInfo(inner_data.wtype);
			if (info == NULL)
				continue;

			grad_weight.Calloc(inner_data.data.count);
			if (!BackwardWeight(i, error_data, output_data, input_data, grad_weight))
			{
				DEBUG_OUTPUT(L"failed calc weight gradient");
				return false;
			}

			if (!m_net_param.optimizer->Update(info->mult_lr, grad_weight.buffer, inner_data))
			{
				DEBUG_OUTPUT(L"failed update weight gradient");
				return false;
			}
		}
	}
	return true;
}

const _NEURO_TENSOR_DATA& HiddenLayerEngine::GetErrorBuffer() const
{
	if (m_error_buffer.GetBuffer() == NULL)
	{
		const_cast<_NEURO_TENSOR_DATA&>(m_error_buffer).AllocLike(GetOutputData());
		m_error_buffer.SetZero();
	}
	return m_error_buffer;
}
