#include "stdafx.h"
#include "cuda_platform.h"

#include "util/np_util.h"

#include "device_launch_parameters.h"
#include <stdio.h>
#include <sstream>

using namespace np::core::cuda;

float dataType::oneval = 1.0;
float dataType::zeroval = 0.0;
const void* dataType::one = static_cast<void *>(&dataType::oneval);
const void* dataType::zero = static_cast<void *>(&dataType::zeroval);

std::vector<_ERROR_INFO> global_lastErrorVector;

const std::vector<_ERROR_INFO>& CudaPlatform::GetErrorVector()
{
	return global_lastErrorVector;
}

std::wstring CudaPlatform::GetErrorString()
{
	std::wstring ret;
	for (size_t i = 0; i < global_lastErrorVector.size(); i++)
	{
		ret += global_lastErrorVector[i].err_string;
		ret += L"\r\n";
	}
	global_lastErrorVector.clear();
	return ret;
}

void CudaPlatform::ClearErrors()
{
	global_lastErrorVector.clear();
}

bool CudaPlatform::CudaErrorCheck(cudaError_t error)
{
	if (error == cudaSuccess)
		return true;

	_ERROR_INFO info;
	info.kind = _ERROR_INFO::_kind::cuda;
	info.error = error;
	swprintf_s(info.err_string, L"%S", cudaGetErrorString(error));

	DEBUG_OUTPUT(info.err_string);

	global_lastErrorVector.push_back(info);
	return false;
}

bool CudaPlatform::CudaLastErrorCheck()
{
	return CudaErrorCheck(cudaGetLastError());
}

const wchar_t* CudaPlatform::CudnnErrorString(cudnnStatus_t status)
{
	switch (status)
	{
	case CUDNN_STATUS_NOT_INITIALIZED:
		return L"CUDNN_STATUS_NOT_INITIALIZED";
	case CUDNN_STATUS_ALLOC_FAILED:
		return L"CUDNN_STATUS_ALLOC_FAILED";
	case CUDNN_STATUS_BAD_PARAM:
		return L"CUDNN_STATUS_BAD_PARAM";
	case CUDNN_STATUS_INTERNAL_ERROR:
		return L"CUDNN_STATUS_INTERNAL_ERROR";
	case CUDNN_STATUS_INVALID_VALUE:
		return L"CUDNN_STATUS_INVALID_VALUE";
	case CUDNN_STATUS_ARCH_MISMATCH:
		return L"CUDNN_STATUS_ARCH_MISMATCH";
	case CUDNN_STATUS_MAPPING_ERROR:
		return L"CUDNN_STATUS_MAPPING_ERROR";
	case CUDNN_STATUS_EXECUTION_FAILED:
		return L"CUDNN_STATUS_EXECUTION_FAILED";
	case CUDNN_STATUS_NOT_SUPPORTED:
		return L"CUDNN_STATUS_NOT_SUPPORTED";
	case CUDNN_STATUS_LICENSE_ERROR:
		return L"CUDNN_STATUS_LICENSE_ERROR";
#if CUDNN_VERSION_MIN(6, 0, 0)
	case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
		return L"CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
#endif
	}
	return NULL;
}

bool CudaPlatform::CudnnErrorCheck(cudnnStatus_t status)
{
	if (status == CUDNN_STATUS_SUCCESS)
		return true;

	const wchar_t* str = CudnnErrorString(status);

	_ERROR_INFO info;
	info.kind = _ERROR_INFO::_kind::cudnn;
	info.error = status;
	if (str != NULL)
		wcscpy_s(info.err_string, str);
	else
		swprintf_s(info.err_string, L"cudnn error. status=%d", status);

	global_lastErrorVector.push_back(info);
	return false;
}

bool CudaPlatform::CublasErrorCheck(cublasStatus_t status)
{
	if (status == CUBLAS_STATUS_SUCCESS)
		return true;

	const wchar_t* str = NULL;
	switch (status)
	{
	case CUBLAS_STATUS_NOT_INITIALIZED:
		str=L"CUBLAS_STATUS_NOT_INITIALIZED";
		break;
	case CUBLAS_STATUS_ALLOC_FAILED:
		str=L"CUBLAS_STATUS_ALLOC_FAILED";
		break;
	case CUBLAS_STATUS_INVALID_VALUE:
		str=L"CUBLAS_STATUS_INVALID_VALUE";
		break;
	case CUBLAS_STATUS_ARCH_MISMATCH:
		str=L"CUBLAS_STATUS_ARCH_MISMATCH";
		break;
	case CUBLAS_STATUS_MAPPING_ERROR:
		str=L"CUBLAS_STATUS_MAPPING_ERROR";
		break;
	case CUBLAS_STATUS_EXECUTION_FAILED:
		str=L"CUBLAS_STATUS_EXECUTION_FAILED";
		break;
	case CUBLAS_STATUS_INTERNAL_ERROR:
		str=L"CUBLAS_STATUS_INTERNAL_ERROR";
		break;
	case CUBLAS_STATUS_NOT_SUPPORTED:
		str=L"CUBLAS_STATUS_NOT_SUPPORTED";
		break;
	case CUBLAS_STATUS_LICENSE_ERROR:
		str=L"CUBLAS_STATUS_LICENSE_ERROR";
		break;
	}
	_ERROR_INFO info;
	info.kind = _ERROR_INFO::_kind::cublas;
	info.error = status;
	if (str != NULL)
		wcscpy_s(info.err_string, str);
	else
		swprintf_s(info.err_string, L"cublas error. status=%d", status);

	global_lastErrorVector.push_back(info);

	return false;
}

bool CudaPlatform::CurandErrorCheck(curandStatus_t status)
{
	if (status == CURAND_STATUS_SUCCESS)
		return true;

	const wchar_t* str = NULL;
	switch (status)
	{
	case CURAND_STATUS_SUCCESS:
		str = L"CURAND_STATUS_SUCCESS";
		break;
	case CURAND_STATUS_VERSION_MISMATCH:
		str = L"CURAND_STATUS_VERSION_MISMATCH";
		break;
	case CURAND_STATUS_NOT_INITIALIZED:
		str = L"CURAND_STATUS_NOT_INITIALIZED";
		break;
	case CURAND_STATUS_ALLOCATION_FAILED:
		str = L"CURAND_STATUS_ALLOCATION_FAILED";
		break;
	case CURAND_STATUS_TYPE_ERROR:
		str = L"CURAND_STATUS_TYPE_ERROR";
		break;
	case CURAND_STATUS_OUT_OF_RANGE:
		str = L"CURAND_STATUS_OUT_OF_RANGE";
		break;
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		str = L"CURAND_STATUS_LENGTH_NOT_MULTIPLE";
		break;
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		str = L"CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
		break;
	case CURAND_STATUS_LAUNCH_FAILURE:
		str = L"CURAND_STATUS_LAUNCH_FAILURE";
		break;
	case CURAND_STATUS_PREEXISTING_FAILURE:
		str = L"CURAND_STATUS_PREEXISTING_FAILURE";
		break;
	case CURAND_STATUS_INITIALIZATION_FAILED:
		str = L"CURAND_STATUS_INITIALIZATION_FAILED";
		break;
	case CURAND_STATUS_ARCH_MISMATCH:
		str = L"CURAND_STATUS_ARCH_MISMATCH";
		break;
	case CURAND_STATUS_INTERNAL_ERROR:
		str = L"CURAND_STATUS_INTERNAL_ERROR";
		break;
	}

	_ERROR_INFO info;
	info.kind = _ERROR_INFO::_kind::curand;
	info.error = status;
	if (str != NULL)
		wcscpy_s(info.err_string, str);
	else
		swprintf_s(info.err_string, L"curand error. status=%d", status);

	global_lastErrorVector.push_back(info);
	return  false;
}

int CudaPlatform::DeviceQuery(cudaDeviceProp* prop)
{
	int device;
	if (cudaSuccess != cudaGetDevice(&device))
		return -1;

	cudaDeviceProp temp;
	if (cudaSuccess != cudaGetDeviceProperties(&temp, device))
		return -1;

	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		return 1;

	if (prop)
		memcpy(prop, &temp, sizeof(cudaDeviceProp));
	return device;
}

bool CudaPlatform::GetMemoryInfo(size_t& free, size_t& total)
{
	return CudaErrorCheck(cudaMemGetInfo(&free, &total));
}

bool CudaPlatform::SetTensor2dDesc(int batch_size, int ch, cudnnTensorDescriptor_t& desc)
{
	return CudnnErrorCheck(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, ch, 1, 1));
}

bool CudaPlatform::SetTensor4dDesc(int batch_size, const tensor::TensorShape& ts, cudnnTensorDescriptor_t& desc)
{
	return CudnnErrorCheck(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, ts.GetChannelCount(), ts.GetHeight(), ts.GetWidth()));
}

CudaInstance* CudaPlatform::CreateInstance()
{
	CudaInstance* instance = new CudaInstance;
	if (!instance)
		return NULL;

	if (!instance->Initialize())
	{
		delete instance;
		return NULL;
	}
	return instance;
}

void CudaPlatform::DestoryInstance(CudaInstance* instance)
{
	delete instance;
}

CudaInstance::CudaInstance()
{
	m_cublas_handle = NULL;
	m_curand_generator = NULL;
}

CudaInstance::~CudaInstance()
{
	if (m_cublas_handle)
		cublasDestroy(m_cublas_handle);
	if (m_curand_generator)
		curandDestroyGenerator(m_curand_generator);
}

bool CudaInstance::Initialize()
{
	DEBUG_OUTPUT(L"start");
	int device = CudaPlatform::DeviceQuery(&m_prop);
	if (device < 0)
	{
		DEBUG_OUTPUT(L"no cuda device");
		return false;
	}

	if (!CudaPlatform::CudaErrorCheck(cudaSetDevice(device)))
	{
		DEBUG_OUTPUT(L"failed set device(%d)", device);
		return false;
	}

	DEBUG_OUTPUT(L"set device[%d", device);
	if (!CudaPlatform::CublasErrorCheck(cublasCreate(&m_cublas_handle)))	// 상당히 오래걸림
	{
		DEBUG_OUTPUT(L"failed cublasCreate");
		return false;
	}
	DEBUG_OUTPUT(L"blas created");

	if (!CudaPlatform::CurandErrorCheck(curandCreateGenerator(&m_curand_generator, CURAND_RNG_PSEUDO_DEFAULT)))
	{
		DEBUG_OUTPUT(L"failed curandCreateGenerator");
		return false;
	}

	DEBUG_OUTPUT(L"end");
	return true;
}

int CudaInstance::GetWarpSize() const
{
	return m_prop.warpSize;
}

int CudaPlatform::GetCudaBlockCount(int N)
{
	int ret = (N + threadsPerBlock - 1) / threadsPerBlock;
	
	return min(ret, 32);
}
