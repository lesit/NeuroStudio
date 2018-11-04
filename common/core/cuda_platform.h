#if !defined(_CUDA_PLATFORM_CUH)
#define _CUDA_PLATFORM_CUH

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

#include "../3rd-party/cudnn-9.1-win10-x64-v7/include/cudnn.h"

#include "../tensor/tensor_shape.h"

#if !defined(INFINITE)
#define INFINITE 0xffffffff
#endif

namespace np
{
	namespace core
	{
		namespace cuda
		{
			struct _ERROR_INFO
			{
				enum class _kind{cuda, cudnn, cublas, curand};
				_kind kind;
				int error;
				wchar_t err_string[128];
			};

			class CudaInstance;

			class CudaPlatform
			{
			public:
				static CudaInstance* CreateInstance();
				static void DestoryInstance(CudaInstance* instance);

				static void ClearErrors();
				static const std::vector<_ERROR_INFO>& GetErrorVector();
				static std::wstring GetErrorString();

				static bool CudaErrorCheck(cudaError_t error);
				static bool CudaLastErrorCheck();
				static const wchar_t* CudnnErrorString(cudnnStatus_t error);
				static bool CudnnErrorCheck(cudnnStatus_t status);
				static bool CublasErrorCheck(cublasStatus_t status);
				static bool CurandErrorCheck(curandStatus_t status);

				static int DeviceQuery(cudaDeviceProp* prop = NULL);

				static bool SetTensor4dDesc(int batch_size, const tensor::TensorShape& ts, cudnnTensorDescriptor_t& desc);
				static bool SetTensor2dDesc(int batch_size, int ch, cudnnTensorDescriptor_t& desc);

				static bool GetMemoryInfo(size_t& free, size_t& total);

				static const int threadsPerBlock = 256;
				static int GetCudaBlockCount(int N);
			};

			class CudaInstance
			{
			public:
				bool Initialize();

				virtual ~CudaInstance();

				int GetWarpSize() const;

				cublasHandle_t cublas_handle() const{
					return m_cublas_handle;
				}

				curandGenerator_t curand_handle() const{
					return m_curand_generator;
				}

				const cudaDeviceProp& GetDeviceProp() const{ return m_prop; }

			protected:
				friend class CudaPlatform;
				CudaInstance();

			private:
				cudaDeviceProp m_prop;

				cublasHandle_t m_cublas_handle;
				curandGenerator_t m_curand_generator;
			};

			class dataType
			{
			public:
				static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
				static float oneval, zeroval;
				static const void *one, *zero;
			};
		}
	}
}

#if !defined(CUDA_KERNEL_LOOP)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
	i < (n); \
	i += blockDim.x * gridDim.x)
#endif

#if !defined(CUDNN_VERSION_MIN)
#define CUDNN_VERSION_MIN(major, minor, patch) \
	(CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))
#endif

#if !defined(CUDNN_CHECK) 
#define CUDNN_CHECK CudaPlatform::CudnnErrorCheck
#endif 

#if !defined(CUDA_POST_KERNEL_CHECK)
#define CUDA_POST_KERNEL_CHECK() CudaPlatform::CudaErrorCheck(cudaPeekAtLastError())
#endif

#endif
