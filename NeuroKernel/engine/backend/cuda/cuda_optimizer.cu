#include "cuda_optimizer.h"

#include "core/cuda_platform.h"

using namespace np::engine::optimizer;
using namespace np::engine::optimizer::cuda;

using namespace np::core::cuda;

Optimizer* Optimizer::CreateInstanceCUDA(_optimizer_type type, const std::vector<neuro_float>& parameters)
{
	switch (type)
	{
	case _optimizer_type::Adagrad:
		return new cuda::AdagradCuda;
	case _optimizer_type::Adam:
		return new cuda::AdamCuda(parameters);
	case _optimizer_type::RMSprop:
		return new cuda::RMSpropCuda;
	default:
		return new cuda::SGDCuda;
	}
}

__global__ void AdagradUpdate(neuro_size_t count, const neuron_weight* dW, neuron_weight* W, neuron_value learn_rate
	, neuron_weight* h, neuron_value eps)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		h[i] += dW[i] * dW[i];
		W[i] -= learn_rate * dW[i] / (std::sqrt(h[i]) + eps);
	}
}

bool AdagradCuda::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* h = weight.GetHistory(0);
	if (!h)
		return false;

	AdagradUpdate << <CudaPlatform::GetCudaBlockCount(weight.data.count), CudaPlatform::threadsPerBlock >> >
		(weight.data.count, dW, weight.data.buffer, learn_rate, h, eps);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void RMSpropUpdate(neuro_size_t count, const neuron_weight* dW, neuron_weight* W, neuron_value learn_rate
	, neuron_weight* h, neuron_value eps, neuron_value decay)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		h[i] = decay * h[i] + (1 - decay) * dW[i] * dW[i];
		W[i] -= learn_rate * dW[i] / std::sqrt(h[i] + eps);
	}
}

bool RMSpropCuda::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* h = weight.GetHistory(0);
	if (!h)
		return false;

	RMSpropUpdate << <CudaPlatform::GetCudaBlockCount(weight.data.count), CudaPlatform::threadsPerBlock >> >
		(weight.data.count, dW, weight.data.buffer, learn_rate, h, eps, decay);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void AdamUpdate(neuro_size_t count, const neuron_weight* dW, neuron_weight* W
	, neuron_weight* mt, neuron_weight* vt
	, neuron_value beta1, neuron_weight beta2
	, neuro_float corrected_local_rate
	, neuron_value eps)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		neuro_float dW_i = dW[i];
		mt[i] = beta1 * mt[i] + (neuro_float(1) - beta1) * dW_i;
		vt[i] = beta2 * vt[i] + (neuro_float(1) - beta2) * dW_i * dW_i;

//		W[i] -= learn_rate * (mt[i] / (neuro_float(1) - b1_t)) / std::sqrt((vt[i] / (neuro_float(1) - b2_t)) + eps);

		W[i] -= corrected_local_rate * mt[i] / (std::sqrt(vt[i]) + eps);
	}
}

bool AdamCuda::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* mt = weight.GetHistory(0);
	neuron_weight* vt = weight.GetHistory(1);
	if (mt == NULL || vt == NULL)
		return false;

	const neuro_float correction = std::sqrt(neuro_float(1) - b2_t) / (neuro_float(1.) - b1_t);

	AdamUpdate << <CudaPlatform::GetCudaBlockCount(weight.data.count), CudaPlatform::threadsPerBlock >> >
		(weight.data.count, dW, weight.data.buffer, mt, vt, beta1, beta2, learn_rate * correction, eps);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void SGDUpdate(neuro_size_t count, const neuron_weight* dW, neuron_weight* W, neuron_value learn_rate
	, neuron_weight* h, neuron_value momentum)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		h[i] = momentum * h[i] + learn_rate * dW[i];
		W[i] -= h[i];
	}
}

bool SGDCuda::Update(neuro_float learn_rate, const neuron_weight* dW, const _LAYER_INNER_DATA& weight)
{
	neuron_weight* h = weight.GetHistory(0);
	if (h == NULL)
		return false;

	SGDUpdate << <CudaPlatform::GetCudaBlockCount(weight.data.count), CudaPlatform::threadsPerBlock >> >
		(weight.data.count, dW, weight.data.buffer, learn_rate, h, momentum);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}
