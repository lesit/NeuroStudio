#include "stdafx.h"

#include "ConcatLayerCudaEngine.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;
using namespace np::core::cuda;

ConcatLayerCudaEngine::ConcatLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: ConcatLayerEngine(net_param, layer)
{
}


ConcatLayerCudaEngine::~ConcatLayerCudaEngine()
{
}

__global__ void Concat(const neuro_u32 N, const neuro_float* in_data,
	const bool forward, const int num_concats, const int concat_size,
	const int output_concat_axis, const int input_concat_axis,
	const int offset_concat_axis, neuro_float* out_data) 
{
	CUDA_KERNEL_LOOP(index, N) 
	{
		const int total_concat_size = concat_size * input_concat_axis;
		const int concat_num = index / total_concat_size;
		const int concat_index = index % total_concat_size;
		const int output_index = concat_index + (concat_num * output_concat_axis + offset_concat_axis) * concat_size;
		if (forward) 
		{
			out_data[output_index] = in_data[index];
		}
		else {
			out_data[index] = in_data[output_index];
		}
	}
}

bool ConcatLayerCudaEngine::Forward(bool bTrain, neuro_u32 batch_size, const _NEURO_TENSOR_DATA& output_buffer)
{
	if (m_input_vector.size() == 0)
		return true;

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
		const int nthreads = input_concat_size * m_num_concats;
		Concat << <CudaPlatform::GetCudaBlockCount(nthreads), CudaPlatform::threadsPerBlock >> >(
			nthreads, input, true, m_num_concats, m_concat_input_size,
			m_output_concat_axis_size, input_concat_axis, offset_concat_axis, output_ptr);

		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed Concat");
			return false;
		}

		offset_concat_axis += input_concat_axis;
	}
	return true;
}

bool ConcatLayerCudaEngine::Backward(neuro_u32 batch_size)
{
	neuron_value* error_ptr = m_error_buffer.data.buffer;

	int offset_concat_axis = 0;
	for (neuro_u32 input_i = 0, input_n = m_input_vector.size(); input_i < input_n; input_i++)
	{
		HiddenLayerEngine* input_engine = (HiddenLayerEngine*)m_input_vector[input_i].engine;
		if (input_engine->GetLayerType() == network::_layer_type::input)
			continue;

		const _NEURO_TENSOR_DATA& input_error = input_engine->GetErrorBuffer();

		const tensor::TensorShape& in_ts = m_input_vector[input_i].engine->GetOutTensorShape();
		const int input_concat_axis = m_concat_axis < 0 ? 1 : in_ts[m_concat_axis];

		const int input_concat_size = input_concat_axis * m_concat_input_size;
		const int nthreads = input_concat_size * m_num_concats;

		if (m_input_vector[input_i].engine->GetLayerType()!=network::_layer_type::input)
		{
			neuron_value* in_err_ptr = input_error.data.buffer;

			Concat << <CudaPlatform::GetCudaBlockCount(nthreads), CudaPlatform::threadsPerBlock >> >(
				nthreads, error_ptr, false, m_num_concats, m_concat_input_size,
				m_output_concat_axis_size, input_concat_axis, offset_concat_axis, in_err_ptr);

			if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
			{
				DEBUG_OUTPUT(L"failed Concat");
				return false;
			}
		}
		offset_concat_axis += input_concat_axis;
	}

	return true;
}
