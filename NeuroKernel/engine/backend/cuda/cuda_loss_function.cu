#include "stdafx.h"

#include "cuda_loss_function.h"
#include "core/cuda_platform.h"

using namespace np::engine::loss;
using namespace np::engine::loss::cuda;
using namespace np::core::cuda;
using namespace std;

LossFunction* LossFunction::CreateInstanceCUDA(core::cuda::CudaInstance* cuda_instance, _loss_type type, bool read_label_for_target)
{
	if (cuda_instance == NULL)
	{
		DEBUG_OUTPUT(L"no cuda instance");
		return NULL;
	}

	switch (type)
	{
	case _loss_type::CrossEntropy:
		return new cuda::CrossEntropy(cuda_instance, read_label_for_target);
	case _loss_type::CrossEntropyMulticlass:
		return new cuda::CrossEntropyMulticlass(cuda_instance, read_label_for_target);
	default:	// case nsas::_loss_type::MSE:
		return new cuda::MSE(cuda_instance, read_label_for_target);
	}
}

CUDALossFunction::CUDALossFunction(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target)
: LossFunction(core::math_device_type::cuda, cuda_instance, read_label_for_target)
{
}

CUDALossFunction::~CUDALossFunction()
{
}

neuron_error CUDALossFunction::CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target)
{
	_NEURO_TENSOR_DATA gpu_loss_buffer(core::math_device_type::cuda, true);
	gpu_loss_buffer.Alloc(batch_size, value_size);
	if (!CalcLossVector(batch_size, value_size, output, target, gpu_loss_buffer.GetBuffer()))
	{
		DEBUG_OUTPUT(L"failed CalcLossVector");
		return -1;
	}

	neuro_float sum_loss;
	if (!sum(gpu_loss_buffer.GetSize(), gpu_loss_buffer.GetBuffer(), sum_loss))
	{
		DEBUG_OUTPUT(L"failed sum");
		return -1;
	}

	return sum_loss;
}

inline __device__ neuro_float GetTargetValueFromLabel(neuro_u32 value_size, neuro_u32 index, const neuro_u32* label_vector)
{
	return label_vector[index / value_size] == (index % value_size) ? 1.f : 0.f;
}

__global__ void MSECalcLoss(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, const bool is_label, neuro_float* loss)
{
	CUDA_KERNEL_LOOP(i, N)
	{
		const neuro_float target_value = is_label ? GetTargetValueFromLabel(value_size, i, (const neuro_u32*)target) : ((neuro_float*)target)[i];

		loss[i] = (output[i] - target_value) * (output[i] - target_value) / 2;
	}
}

bool MSE::CalcLossVector(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* loss)
{
	const neuro_u32 N = batch_size * value_size;

	MSECalcLoss << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >
		(N, value_size, output, target, m_read_label_for_target, loss);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void MSECalcDiff(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, const bool is_label, neuro_float scale, neuro_float* diff)
{
	CUDA_KERNEL_LOOP(i, N)
	{
		const neuro_float target_value = is_label ? GetTargetValueFromLabel(value_size, i, (const neuro_u32*)target) : ((neuro_float*)target)[i];

		diff[i] = scale * (output[i] - target_value);
	}
}

bool MSE::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	const neuron_error scale = normalize_factor(batch_size);
	const neuro_u32 N = batch_size * value_size;

	MSECalcDiff << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >
		(N, value_size, output, target, m_read_label_for_target, scale, diff);

#if 0//defined(_DEBUG)
	void* temp=malloc(batch_size*value_size*4);

	core::CPU_MemoryManager cpu;

	DEBUG_OUTPUT(L"output");
	cpu.Memcpy(temp, output, batch_size * value_size * 4, core::math_device_type::cuda);
	NP_Util::DebugOutputValues((neuro_float*)temp, batch_size*value_size, 10);

	DEBUG_OUTPUT(L"diff");
	cpu.Memcpy(temp, diff, batch_size * value_size * 4, core::math_device_type::cuda);
	NP_Util::DebugOutputValues((neuro_float*)temp, batch_size*value_size, 10);

	DEBUG_OUTPUT(L"label");
	cpu.Memcpy(temp, target, batch_size * 4, core::math_device_type::cuda);
	NP_Util::DebugOutputValues((neuro_u32*)temp, batch_size, 10);

	free(temp);
#endif

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

//#define _LOG_THRESHOLD 1e-20

// 흠.. cross entropy는 좀더 확인해봐야함...
CrossEntropy::CrossEntropy(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target)
: CUDALossFunction(cuda_instance, read_label_for_target)
{
}

__global__ void CrossEntropyLoss(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, const bool is_label, neuro_float* loss)
{
	// output가 0이거나 1일때 문제가 되서 FLT_MIN 사용
	CUDA_KERNEL_LOOP(i, N)
	{
		const neuro_float target_value = is_label ? GetTargetValueFromLabel(value_size, i, (const neuro_u32*)target) : ((neuro_float*)target)[i];

		loss[i] = -target_value * log(max(output[i], FLT_MIN)) - (neuro_float(1) - target_value) * log(max(neuro_float(1) - output[i], FLT_MIN));
	}
}

bool CrossEntropy::CalcLossVector(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* loss)
{
	CrossEntropyLoss << <CudaPlatform::GetCudaBlockCount(batch_size*value_size), CudaPlatform::threadsPerBlock >> >
		(batch_size*value_size, value_size, output, target, m_read_label_for_target, loss);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

__global__ void CrossEntropyDiff(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, const bool is_label, neuro_float scale, neuro_float* diff)
{
	CUDA_KERNEL_LOOP(i, N)
	{
		const neuro_float target_value = is_label ? GetTargetValueFromLabel(value_size, i, (const neuro_u32*)target) : ((neuro_float*)target)[i];

		neuro_float prob = max(output[i], FLT_MIN);

		diff[i] = scale * target_value / prob;
	}
}

bool CrossEntropy::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	const neuron_error scale = -normalize_factor(batch_size);
	const neuro_u32 N = batch_size * value_size;

	CrossEntropyDiff << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >
		(N, value_size, output, target, m_read_label_for_target, scale, diff);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}

CrossEntropyMulticlass::CrossEntropyMulticlass(core::cuda::CudaInstance* cuda_instance, bool read_label_for_target)
: CUDALossFunction(cuda_instance, read_label_for_target)
{
}

__global__ void CrossEntropyMulticlassLoss(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, bool is_label, neuro_float* loss)
{
	// output가 0일때 문제가 되서 FLT_MIN 사용
	CUDA_KERNEL_LOOP(sample, N)
	{
		neuro_u32 label = 0;
		if (is_label)
		{
			label = ((neuro_u32*)target)[sample];
		}
		else
		{
			const neuro_float* target_p = (neuro_float*)target + sample*value_size;

			for (neuro_u32 t_index = 1; t_index < value_size; t_index++)
			{
				if (target_p[t_index]>target_p[label])
					label = t_index;
			}
		}

		loss[sample] = -log(max(output[sample*value_size + label], FLT_MIN));
	}
}

// cross-entropy loss function for multi-class classification
neuron_error CrossEntropyMulticlass::CalcLoss(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target)
{
	_NEURO_TENSOR_DATA gpu_loss_buffer(core::math_device_type::cuda, true);
	gpu_loss_buffer.Alloc(batch_size, 1);

	CrossEntropyMulticlassLoss << <CudaPlatform::GetCudaBlockCount(batch_size), CudaPlatform::threadsPerBlock >> >
		(batch_size, value_size, output, target, m_read_label_for_target, gpu_loss_buffer.GetBuffer());

	if(!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
	{
		DEBUG_OUTPUT(L"failed CrossEntropyMulticlassCalcLoss");
		return -1;
	}

	neuro_float sum_loss;
	if (!sum(gpu_loss_buffer.GetSize(), gpu_loss_buffer.GetBuffer(), sum_loss))
	{
		DEBUG_OUTPUT(L"failed sum");
		return -1;
	}

	return sum_loss;
}

__global__ void CrossEntropyMulticlassDiff(neuro_u32 N, neuro_u32 value_size, const neuro_float* output, const void* target, bool is_label, neuro_float scale, neuro_float* diff)
{
	CUDA_KERNEL_LOOP(sample, N)
	{
		neuro_u32 label = 0;
		if (is_label)
		{
			label = ((neuro_u32*)target)[sample];
		}
		else
		{
			const neuro_float* target_p = (neuro_float*)target + sample*value_size;

			for (neuro_u32 t_index = 1; t_index < value_size; t_index++)
			{
				if (target_p[t_index]>target_p[label])
					label = t_index;
			}
		}
		neuro_float prob = max(output[sample*value_size + label], FLT_MIN);

		diff[sample*value_size + label] = scale / prob;
	}
}

bool CrossEntropyMulticlass::CalcDiff(neuro_u32 batch_size, neuro_u32 value_size, const neuro_float* output, const void* target, neuro_float* diff)
{
	neuron_error scale = - normalize_factor(batch_size);

	CrossEntropyMulticlassDiff << <CudaPlatform::GetCudaBlockCount(batch_size), CudaPlatform::threadsPerBlock >> >
		(batch_size, value_size, output, target, m_read_label_for_target, scale, diff);

	return CudaPlatform::CudaErrorCheck(cudaPeekAtLastError());
}
