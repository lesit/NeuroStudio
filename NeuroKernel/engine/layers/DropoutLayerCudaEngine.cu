#include "DropoutLayerCudaEngine.h"

#include "core/cuda_platform.h"

using namespace np::engine;
using namespace np::engine::layers;
using namespace np::core::cuda;

DropoutLayerCudaEngine::DropoutLayerCudaEngine(const NetworkParameter& net_param, const network::HiddenLayer& layer)
: DropoutLayerEngine(net_param, layer)
{
}

DropoutLayerCudaEngine::~DropoutLayerCudaEngine()
{
	m_mask.Dealloc();
}

__global__ void DropoutForward(const neuro_u32 n, const float* input,
	const neuro_u32* mask, const neuro_u32 threshold, const float scale,
	float* output) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		output[index] = input[index] * (mask[index] > threshold) * scale;
	}
}

bool DropoutLayerCudaEngine::ForwardData(bool bTrain, const _NEURO_TENSOR_DATA& input_data, const _NEURO_TENSOR_DATA& output_data)
{
	if (output_data.GetSize() != input_data.GetSize())
	{
		DEBUG_OUTPUT(L"output data's size is not equal with input");
		return false;
	}
	if (bTrain)
	{
		if (!m_net_param.cuda_instance)
		{
			return false;
		}

		if (!CudaPlatform::CurandErrorCheck(curandGenerate(m_net_param.cuda_instance->curand_handle()
			, m_mask.GetBuffer(), m_mask.GetSize())))
		{
			DEBUG_OUTPUT(L"failed curandGenerate. %s", CudaPlatform::GetErrorString().c_str());
			return false;
		}

		// set thresholds
		// NOLINT_NEXT_LINE(whitespace/operators)
		DropoutForward << <CudaPlatform::GetCudaBlockCount(input_data.GetSize()), CudaPlatform::threadsPerBlock >> >(
			input_data.GetSize(), input_data.GetBuffer(), m_mask.GetBuffer(), m_uint_threshold, m_dropout_scale, output_data.GetBuffer());

		if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
		{
			DEBUG_OUTPUT(L"failed DropoutForward");
			return false;
		}
	}
	else 
	{
		if (!output_data.CopyFrom(input_data))
		{
			return false;
		}
	}

	return true;
}

__global__ void DropoutBackward(const neuro_u32 n, const float* cur_error,
	const unsigned int* mask, const unsigned int threshold, const float scale,
	float* in_error) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		in_error[index] = cur_error[index] * scale * (mask[index] > threshold);
	}
}

bool DropoutLayerCudaEngine::BackwardError(const _NEURO_TENSOR_DATA& output_data
	, const _NEURO_TENSOR_DATA& current_error
	, const _NEURO_TENSOR_DATA& input_data
	, const _NEURO_TENSOR_DATA& input_error)
{
	if (input_error.GetSize() != current_error.GetSize())
	{
		DEBUG_OUTPUT(L"output data's size is not equal with input");
		return false;
	}

	// 당연히 train이므로..
	if (m_mask.GetBuffer() == NULL)
	{
		DEBUG_OUTPUT(L"size of mask is not equal with output");
		return false;
	}

	const neuro_size_t count = input_error.GetSize();

	// NOLINT_NEXT_LINE(whitespace/operators)
	DropoutBackward << <CudaPlatform::GetCudaBlockCount(count), CudaPlatform::threadsPerBlock >> >(
		count, current_error.GetBuffer(), m_mask.GetBuffer(), m_uint_threshold, m_dropout_scale, input_error.GetBuffer());

	if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
	{
		DEBUG_OUTPUT(L"failed DropoutBackward");
		return false;
	}
	return true;
}
