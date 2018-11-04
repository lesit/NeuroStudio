#include "stdafx.h"

#include "ConcatLayerEngine.h"
#include "ConcatLayerCudaEngine.h"

#include "../../network/ConcatLayerConfigure.h"

using namespace np::engine;
using namespace np::engine::layers;

ConcatLayerEngine* ConcatLayerEngine::CreateInstance(const NetworkParameter& net_param, const network::HiddenLayer& layer)
{
	if (net_param.run_pdtype == core::math_device_type::cuda)
		return new ConcatLayerCudaEngine(net_param, layer);
	else
		return new ConcatLayerEngine(net_param, layer);
}

ConcatLayerEngine::ConcatLayerEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: HiddenLayerEngine(net_param, layer)
{
	m_num_concats = 0;
	m_concat_input_size = 0;

	m_output_concat_axis_size = 0;
}


ConcatLayerEngine::~ConcatLayerEngine()
{
}

bool ConcatLayerEngine::OnInitialized()
{
	m_concat_axis = -1;
	return true;
}

bool ConcatLayerEngine::OnOutputBufferInitialized(const _NEURO_TENSOR_DATA& buf)
{
	np::network::_CONCAT_INFO concat_info = np::network::ConcatLayerConfigure::GetConcatInfo((const HiddenLayer&)m_layer);
	if (concat_info.toTensor() != m_out_ts)
	{
		DEBUG_OUTPUT(L"the tensor is strange!");
		return false;
	}

	m_concat_axis = concat_info.concat_axis;
	if (concat_info.concat_axis<0)
		m_num_concats = buf.batch_size;
	else
		m_num_concats = buf.batch_size * concat_info.join_ts.GetTensorSize();

	// 만약 concat_axis가 0보다 작다면 time이 axis가 된다.
	m_output_concat_axis_size = concat_info.concat_axis_size;

	// 만약 axis 이후에 불일치한다면 m_concat_input_size 은 1이 될 것이다.
	m_concat_input_size = concat_info.concat_ts.GetDimSize();
	return true;
}

bool ConcatLayerEngine::Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output_buffer)
{
	const core::MemoryManager& mm = output_buffer.data.mm;
	neuron_value* output_ptr = output_buffer.data.buffer;

	int offset_concat_axis = 0;
	for (neuro_u32 input_i = 0, input_n = m_input_vector.size(); input_i < input_n; input_i++)
	{
		_NEURO_TENSOR_DATA input_data;
		if (!GetInputData(m_input_vector[input_i], input_data))
		{
			DEBUG_OUTPUT(L"no input data");
			return false;
		}

		const neuron_value* input = input_data.data.buffer;
		const tensor::TensorShape& in_ts = m_input_vector[input_i].engine->GetOutTensorShape();
		const int input_concat_axis = m_concat_axis < 0 ? 1 : in_ts[m_concat_axis];
		const int input_concat_size = input_concat_axis * m_concat_input_size;

		for (neuro_u32 i = 0; i < m_num_concats; i++)
		{
			mm.Memcpy(output_ptr + (i * m_output_concat_axis_size + offset_concat_axis) * m_concat_input_size
				, input + i * input_concat_size
				, input_concat_size * sizeof(neuron_value), mm);	// memcpy 할땐 항상 sizeof 에 신경 좀 쓰자!!
		}
		offset_concat_axis += input_concat_axis;
	}

	return true;
}

bool ConcatLayerEngine::Backward(neuro_u32 batch_size)
{
	const core::MemoryManager& mm = m_error_buffer.data.mm;
	neuron_value* error_ptr = m_error_buffer.data.buffer;

	int offset_concat_axis = 0;
	for (neuro_u32 input_i = 0, input_n = m_input_vector.size(); input_i < input_n; input_i++)
	{
		HiddenLayerEngine* input_engine = (HiddenLayerEngine*) m_input_vector[input_i].engine;
		if (input_engine->GetLayerType() == network::_layer_type::input)
			continue;

		const _NEURO_TENSOR_DATA& input_error = input_engine->GetErrorBuffer();

		const tensor::TensorShape& in_ts = m_input_vector[input_i].engine->GetOutTensorShape();
		const int input_concat_axis = m_concat_axis < 0 ? 1 : in_ts[m_concat_axis];
		const int input_concat_size = input_concat_axis * m_concat_input_size;
		if (m_input_vector[input_i].engine->GetLayerType() != network::_layer_type::input)
		{
			neuron_value* in_err_ptr = input_error.data.buffer;

			for (neuro_u32 i = 0; i < m_num_concats; i++)
			{
				mm.Memcpy(in_err_ptr + i * input_concat_size
					, error_ptr + (i * m_output_concat_axis_size + offset_concat_axis) * m_concat_input_size
					, input_concat_size * sizeof(neuron_value), mm);
			}
		}
		offset_concat_axis += input_concat_axis;
	}

	return true;
}
