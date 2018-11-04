#include "stdafx.h"

#include "MemoryManager.h"
#include "util/np_util.h"

#include "cuda_platform.h"

using namespace np;
using namespace np::core;
using namespace np::core::cuda;

CPU_MemoryManager cpuInstance;
CUDA_MemoryManager cudaInstance;
MemoryManager& MemoryManager::GetManager(math_device_type type)
{
	if (type == math_device_type::cuda)
		return cudaInstance;
	return cpuInstance;
}

bool CPU_MemoryManager::GetMemoryInfo(neuro_size_t& free, neuro_size_t& total) const
{
	return false;
}

void* CPU_MemoryManager::Alloc(void* old, neuro_size_t old_size, neuro_size_t new_size) const
{
#ifdef _DEBUG
	if (new_size == 400000)
		int a = 0;
	if (new_size == 31360000)
		int a = 0;
#endif
	if (old)
		return realloc(old, new_size);
	else
		return malloc(new_size);
}
bool CPU_MemoryManager::Dealloc(void* old) const
{
	if (old != 0)
		free(old);
	return true;
}

bool CPU_MemoryManager::SetZero(void* data, neuro_size_t size) const
{
	if (data == NULL)
		return false;

	memset(data, 0, size);
	return true;
}

bool CPU_MemoryManager::DataSet(neuro_u32* data, neuro_u32 value, neuro_size_t count) const
{
	if (data == NULL)
		return false;

	if (value == 0)
	{
		memset(data, 0, sizeof(neuro_u32)* count);
		return true;
	}
	for (neuro_size_t i = 0; i < count; ++i)
		data[i] = value;

	return true;
}

bool CPU_MemoryManager::DataSet(neuron_value* data, neuron_value value, neuro_size_t count) const
{
	if (data == NULL)
		return false;

	if (value == 0)
	{
		memset(data, 0, sizeof(neuron_value)* count);
		return true;
	}
	for (neuro_size_t i = 0; i < count; ++i)
		data[i] = value;

	return true;
}

bool CPU_MemoryManager::Memcpy(void* target, const void* source, neuro_size_t size, math_device_type srctype) const
{
	if (target == NULL || source == NULL)
		return false;

	if (srctype == math_device_type::cpu)
	{
		memcpy(target, source, size);
		return true;
	}

	if(!CudaPlatform::CudaErrorCheck(cudaMemcpy(target, source, size, cudaMemcpyDeviceToHost)))
	{
		DEBUG_OUTPUT(L"failed. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	return true;
}

bool CUDA_MemoryManager::GetMemoryInfo(neuro_size_t& free, neuro_size_t& total) const
{
	return core::cuda::CudaPlatform::GetMemoryInfo(free, total);
}

void* CUDA_MemoryManager::Alloc(void* old, neuro_size_t old_size, neuro_size_t new_size) const
{
#ifdef _DEBUG
	if (new_size == 2304000)
		int a = 0;
	if (new_size == 512000)
		int a = 0;
#endif

	if (old != NULL)
	{
		if (old_size == new_size)
			return old;

		if (!CudaPlatform::CudaErrorCheck(cudaFree(old)))
		{
			DEBUG_OUTPUT(L"failed. cudaFree %s", CudaPlatform::GetErrorString().c_str());
			return NULL;
		}
	}
#ifdef _DEBUG
	if (new_size >= 499 * 20 * 24 * 24 * sizeof(neuron_value))
		int a = 0;
#endif
	size_t free, total;
	core::cuda::CudaPlatform::GetMemoryInfo(free, total);

	void* newMem=NULL;
	if (!CudaPlatform::CudaErrorCheck(cudaMalloc(&newMem, new_size)))
	{
		DEBUG_OUTPUT(L"failed to cudaMalloc{%llu]. free[%lld], total[%lld] of gpu memory. %s", new_size, free, total, CudaPlatform::GetErrorString().c_str());
		return NULL;
	}

	return newMem;
}

bool CUDA_MemoryManager::Dealloc(void* old) const
{
	if (old == NULL)
		return true;

	if (!CudaPlatform::CudaErrorCheck(cudaFree(old)))
	{
		DEBUG_OUTPUT(L"failed. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	return true;
}

bool CUDA_MemoryManager::SetZero(void* data, neuro_size_t size) const
{
	if (data == NULL)
		return false;

	if(!CudaPlatform::CudaErrorCheck(cudaMemset(data, 0, size)))
	{
		DEBUG_OUTPUT(L"failed. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	return true;
}

bool CUDA_MemoryManager::DataSet(neuro_u32* data, neuro_u32 value, neuro_size_t count) const
{
	if (data == NULL)
		return false;

	neuro_u32* cpu_data = (neuro_u32*)malloc(count * sizeof(neuro_u32));
	if (!cpu_data)
		return false;

	CPU_MemoryManager cpu_m;
	cpu_m.DataSet(cpu_data, value, count);

	bool bRet = Memcpy(data, cpu_data, count * sizeof(neuro_u32), cpu_m.GetType());

	free(cpu_data);

	return bRet;
}

bool CUDA_MemoryManager::DataSet(neuron_value* data, neuron_value value, neuro_size_t count) const
{
	if (data == NULL)
		return false;

	neuron_value* cpu_data = (neuron_value*)malloc(count * sizeof(neuron_value));
	if (!cpu_data)
		return false;

	CPU_MemoryManager cpu_m;
	cpu_m.DataSet(cpu_data, value, count);

	bool bRet = Memcpy(data, cpu_data, count * sizeof(neuron_value), cpu_m.GetType());

	free(cpu_data);

	return bRet;
}


bool CUDA_MemoryManager::Memcpy(void* target, const void* source, neuro_size_t size, math_device_type srctype) const
{
	if (target == NULL || source == NULL)
		return false;

	cudaMemcpyKind kind;
	switch (srctype)
	{
	case math_device_type::cuda:
		kind = cudaMemcpyDeviceToDevice;
		break;
	case math_device_type::cpu:
		kind = cudaMemcpyHostToDevice;
		break;
	default:
		return false;
	}
	
	if (!CudaPlatform::CudaErrorCheck(cudaMemcpy(target, source, size, kind)))
	{
		DEBUG_OUTPUT(L"failed. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}
	return true;
}
