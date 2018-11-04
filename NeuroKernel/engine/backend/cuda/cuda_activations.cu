#include "cuda_activations.h"
#include "core/cuda_platform.h"

using namespace np::engine::activation::cuda;

namespace np
{
	namespace engine
	{
		namespace activation
		{
			ActivationFunction* ActivationFunction::CreateInstanceCUDA(network::_activation_type type)
			{
				switch (type)
				{
				case network::_activation_type::sigmoid:
				case network::_activation_type::reLU:
				case network::_activation_type::tahn:
					return new CuDNNActivation(type);
				case network::_activation_type::leakyReLU:
					return new CudaLeakyReLuActivation;
				case network::_activation_type::eLU:
					return new CudaELuActivation;
				case network::_activation_type::softmax:
					return new CudaSoftmaxActivation();
				}
				return NULL;
			}
		}
	}
}

using namespace np::core::cuda;

CuDNNActivation::CuDNNActivation(network::_activation_type type)
: m_type(type)
{
	m_cudnn_handle = NULL;
	m_ts_desc = NULL;
	m_activ_desc = NULL;

	if (!CUDNN_CHECK(cudnnCreate(&m_cudnn_handle)))
		return;

	cudnnActivationMode_t mode = CUDNN_ACTIVATION_TANH;
	switch (type)
	{
	case network::_activation_type::sigmoid:
		mode = CUDNN_ACTIVATION_SIGMOID;
		break;
	case network::_activation_type::reLU:
		mode = CUDNN_ACTIVATION_RELU;
		break;
	case network::_activation_type::tahn:
		mode = CUDNN_ACTIVATION_TANH;
		break;
	default:
		return;
	}

	if (!CUDNN_CHECK(cudnnCreateActivationDescriptor(&m_activ_desc)))
	{
		DEBUG_OUTPUT(L"failed create activation");
		return;
	}

	if (!CUDNN_CHECK(cudnnSetActivationDescriptor(m_activ_desc, mode, CUDNN_PROPAGATE_NAN, float_t(0))))
	{
		DEBUG_OUTPUT(L"failed cudnnSetActivationDescriptor");
		return;
	}

	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_ts_desc)))
	{
		DEBUG_OUTPUT(L"failed cudnnCreateTensorDescriptor");
		return;
	}
}

CuDNNActivation::~CuDNNActivation()
{
	if (m_activ_desc != NULL)
	{
		if (!CUDNN_CHECK(cudnnDestroyActivationDescriptor(m_activ_desc)))
		{
			DEBUG_OUTPUT(L"failed destory activation");
		}
	}
	if (!CUDNN_CHECK(cudnnDestroyTensorDescriptor(m_ts_desc)))
	{
		DEBUG_OUTPUT(L"failed destory tensor desc");
	}

	if (m_cudnn_handle)
		cudnnDestroy(m_cudnn_handle);
}

bool CuDNNActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	if (m_ts_desc == NULL)
		return false;

	if (!CudaPlatform::SetTensor2dDesc(value.GetBatchSize(), value.value_size, m_ts_desc))
	{
		DEBUG_OUTPUT(L"failed set tensor desc");
		return false;
	}

	if (!CUDNN_CHECK(cudnnActivationForward(m_cudnn_handle
		, m_activ_desc
		, &dataType::oneval
		, m_ts_desc, value.GetBuffer()
		, &dataType::zeroval
		, m_ts_desc, value.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnActivationForward");
		return false;
	}

	return true;
}

bool CuDNNActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	/*	입력 값은 단지 ReLU에서만 사용하며 0보다 큰지를 비교하는데 사용된다.
		ReLU에서의 output은 input값이 0보다 클경우 똑같은 값이고 0이거나 작을경우 0으로 처리하므로
		output을 그대로 input으로 사용해도 무방하다.
		애매하면 leaky relu를 사용하면 된다.
	*/
	const neuron_value* in = out.GetBuffer();

	if (!CUDNN_CHECK(cudnnActivationBackward(m_cudnn_handle
		, m_activ_desc
		, &dataType::oneval
		, m_ts_desc, out.GetBuffer()
		, m_ts_desc, error.GetBuffer()
		, m_ts_desc, in	
		, &dataType::zeroval
		, m_ts_desc, error.GetBuffer())))	// 만약 이게 문제되면 delta 버퍼를 다시 할당하여 계산한후 다시 복사하자!
	{
		DEBUG_OUTPUT(L"failed cudnnActivationBackward");
		return false;
	}

	return true;
}

__global__ void ReLUForward(const int n, float* value, float negative_slope)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		value[index] = value[index] > 0 ? value[index] : value[index] * negative_slope;
	}
}

bool CudaLeakyReLuActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	ReLUForward << <CudaPlatform::GetCudaBlockCount(value.GetSize()), CudaPlatform::threadsPerBlock >> >(value.GetSize(), value.GetBuffer(), m_relu_negative_slope);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void ReLUBackward(const int n, const float* value, float* error, float negative_slope) {
	CUDA_KERNEL_LOOP(index, n) 
	{
		error[index] *= value[index] > neuron_value(0) ? neuron_value(1) : negative_slope;
	}
}

bool CudaLeakyReLuActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	ReLUBackward << <CudaPlatform::GetCudaBlockCount(error.GetSize()), CudaPlatform::threadsPerBlock >> >(error.GetSize(), out.GetBuffer(), error.GetBuffer(), m_relu_negative_slope);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void ELUForward(const int n, float* value, float alpha)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		value[index] = value[index] < 0 ? alpha * (exp(value[index]) - 1) : value[index];
	}
}

bool CudaELuActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	ELUForward << <CudaPlatform::GetCudaBlockCount(value.GetSize()), CudaPlatform::threadsPerBlock >> >(value.GetSize(), value.GetBuffer(), m_elu_alpha);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void ELUBackward(const int n, const float* value, float* error, float alpha)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		error[index] *= value[index] > neuron_value(0) ? neuron_value(1) : (alpha + value[index]);
	}
}
bool CudaELuActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	ELUBackward << <CudaPlatform::GetCudaBlockCount(error.GetSize()), CudaPlatform::threadsPerBlock >> >(error.GetSize(), out.GetBuffer(), error.GetBuffer(), m_elu_alpha);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

#include "core/MemoryManager.h"

/*
SoftmaxActivation::SoftmaxActivation()
{
	m_cudnn_handle = NULL;
	m_in_desc = m_out_desc = NULL;

	if (!CUDNN_CHECK(cudnnCreate(&m_cudnn_handle)))
		return;

	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_in_desc)))
		DEBUG_OUTPUT(L"failed cudnnCreateTensorDescriptor");
	if (!CUDNN_CHECK(cudnnCreateTensorDescriptor(&m_out_desc)))
		DEBUG_OUTPUT(L"failed cudnnCreateTensorDescriptor");
}

SoftmaxActivation::~SoftmaxActivation()
{
	if (m_cudnn_handle)
		CUDNN_CHECK(cudnnDestroy(m_cudnn_handle));
	if (m_in_desc)
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(m_in_desc));
	if (m_out_desc)
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(m_out_desc));
}

bool SoftmaxActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	if (m_in_desc == NULL || m_out_desc == NULL)
		return false;

	if (!CudaPlatform::SetTensor2dDesc(value.GetBatchSize(), value.value_size, m_in_desc))
	{
		DEBUG_OUTPUT(L"failed set tensor desc");
		return false;
	}
	if (!CudaPlatform::SetTensor2dDesc(value.GetBatchSize(), value.value_size, m_out_desc))
	{
		DEBUG_OUTPUT(L"failed set tensor desc");
		return false;
	}

	core::CUDA_MemoryManager mm;
	mm.Memcpy(add_buf.GetBuffer(), value.GetBuffer(), value.GetSize(), mm.GetType());

	if (!CUDNN_CHECK(
		cudnnSoftmaxForward(m_cudnn_handle
		, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL
		, dataType::one
		, m_in_desc, add_buf.GetBuffer()
		, dataType::zero
		, m_out_desc, value.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnSoftmaxForward");
		return false;
	}
	return true;
}

bool SoftmaxActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	if (!CUDNN_CHECK(
		cudnnSoftmaxBackward(m_cudnn_handle
		, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE //CUDNN_SOFTMAX_MODE_CHANNEL
		, dataType::one
		, m_out_desc, out.GetBuffer()
		, m_out_desc, error.GetBuffer()
		, dataType::zero
		, m_in_desc, delta.GetBuffer())))
	{
		DEBUG_OUTPUT(L"failed cudnnSoftmaxBackward");
		return false;
	}
	return true;
}
*/

CudaSoftmaxActivation::CudaSoftmaxActivation()
{
}

CudaSoftmaxActivation::~CudaSoftmaxActivation()
{
}

__global__ void kernel_max(const int num, const int value_size, const neuro_float* data, neuro_float* out) 
{
	CUDA_KERNEL_LOOP(index, num)
	{
		neuro_float maxval = -FLT_MAX;
		for (int c = 0; c < value_size; ++c) 
		{
			if (maxval < data[(index * value_size + c)])
				maxval = data[(index * value_size + c)];
		}
		out[index] = maxval;
	}
}

__global__ void kernel_subtract_exp(const int count, const int value_size, const neuro_float* factor, neuro_float* data) 
{
	CUDA_KERNEL_LOOP(index, count) 
	{
		data[index] -= factor[index / value_size];
		data[index] = exp(data[index]);
	}
}

__global__ void kernel_sum(const int num, const int value_size, const neuro_float* data, neuro_float* channel_sum) {
	CUDA_KERNEL_LOOP(index, num)
	{
		neuro_float sum = 0;
		for (int c = 0; c < value_size; ++c)
		{
			sum += data[(index * value_size + c)];
		}
		channel_sum[index] = sum;
	}
}

__global__ void kernel_div(const int count, const int value_size, const neuro_float* sum, neuro_float* data) 
{
	CUDA_KERNEL_LOOP(index, count) 
	{
		data[index] /= sum[index / value_size];
	}
}

__global__ void kernel_dot(const int num, const int value_size, const neuro_float* data_1, const neuro_float* data_2,
	neuro_float* dot_out) 
{
	CUDA_KERNEL_LOOP(index, num) 
	{
		neuro_float dot = 0;
		for (int c = 0; c < value_size; ++c) 
		{
			dot += (data_1[(index * value_size + c)] * data_2[(index * value_size + c)]);
		}
		dot_out[index] = dot;
	}
}

__global__ void softmax_subtract_mult(const int count, const int value_size, const neuro_float* sub_factor, const neuro_float* mult_factor, neuro_float* data)
{
	CUDA_KERNEL_LOOP(index, count)
	{
		data[index] = (data[index] - sub_factor[index / value_size]) * mult_factor[index];
	}
}

bool CudaSoftmaxActivation::ForwardActivations(const _NEURO_TENSOR_DATA& value)
{
	// We need to subtract the max to avoid numerical issues, compute the exp,
	// and then normalize.
	// compute max
	// NOLINT_NEXT_LINE(whitespace/operators)
	int all_blocks = CudaPlatform::GetCudaBlockCount(value.GetSize());
	int batch_blocks = CudaPlatform::GetCudaBlockCount(value.GetBatchSize());

	_NEURO_TENSOR_DATA temp(core::math_device_type::cuda, true);
	if(!temp.AllocLike(value))
	{
		DEBUG_OUTPUT(L"failed alloc temp");
		return false;
	}

	// 흠.. 원래는 softmax를 위해서 time을 channel로 사용해야함

	kernel_max << <batch_blocks, CudaPlatform::threadsPerBlock >> >(value.GetBatchSize(), value.value_size, value.GetBuffer(), temp.GetBuffer());
	// subtract
	// exponentiate
	kernel_subtract_exp << <all_blocks, CudaPlatform::threadsPerBlock >> >(value.GetSize(), value.value_size, temp.GetBuffer(), value.GetBuffer());

	// sum after exp
	kernel_sum << <batch_blocks, CudaPlatform::threadsPerBlock >> >(value.GetBatchSize(), value.value_size, value.GetBuffer(), temp.GetBuffer());
	// divide
	kernel_div << <all_blocks, CudaPlatform::threadsPerBlock >> >(value.GetSize(), value.value_size, temp.GetBuffer(), value.GetBuffer());

	return true;
}

bool CudaSoftmaxActivation::BackwardActivations(const _NEURO_TENSOR_DATA& out, const _NEURO_TENSOR_DATA& error)
{
	int all_blocks = CudaPlatform::GetCudaBlockCount(error.GetSize());
	int batch_blocks = CudaPlatform::GetCudaBlockCount(error.GetBatchSize());

	_NEURO_TENSOR_DATA dot(core::math_device_type::cuda, true);
	if (!dot.AllocLike(out))
	{
		DEBUG_OUTPUT(L"failed alloc dot");
		return false;
	}

	// Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
	kernel_dot << <batch_blocks, CudaPlatform::threadsPerBlock >> >(error.GetBatchSize(), error.value_size, error.GetBuffer(), out.GetBuffer(), dot.GetBuffer());

	softmax_subtract_mult << <all_blocks, CudaPlatform::threadsPerBlock >> >(error.GetSize(), error.value_size, dot.GetBuffer(), out.GetBuffer(), error.GetBuffer());

	return true;
}
