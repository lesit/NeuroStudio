#if !defined(_MATH_GPU_CORE_H)
#define _MATH_GPU_CORE_H

#include "MathCoreApi.h"
#include "cuda_platform.h"
namespace np
{
	namespace core
	{
		class MathGpuCore : public AbstractMathCore
		{
		public:
			MathGpuCore();

			bool Set(cuda::CudaInstance* _cuda_instance)
			{
				m_cuda_instance = _cuda_instance;
				if (m_cuda_instance)
					m_cublas_handle = m_cuda_instance->cublas_handle();

				return m_cuda_instance!=NULL;
			}

		private:
			cuda::CudaInstance* m_cuda_instance;
			cublasHandle_t m_cublas_handle;

		public:
			bool gemm(const CBLAS_TRANSPOSE TransA,
				const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
				const neuro_float alpha, const neuro_float* A, const neuro_float* B, const neuro_float beta,
				neuro_float* C) override;

			bool gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
				, const neuro_float alpha, const neuro_float* A, const neuro_float* x
				, const neuro_float beta, neuro_float* y) override;

			bool axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y) override;

			bool scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X) override;

			bool axpby(const neuro_u32 N
				, const neuro_float alpha, const neuro_float* X
				, const neuro_float beta, neuro_float* Y) override;

			bool sum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) override;
			bool asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) override;

			bool add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y) override;
			bool powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y) override;
			bool sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) override;
			bool mul(const neuro_u32 n, const neuro_float* a, const neuro_float* b, neuro_float* y) override;
			bool div(const neuro_u32 n, const neuro_float* a, const neuro_float* b, neuro_float* y) override;

			bool dot(const neuro_u32 n, const neuro_float* a, const neuro_float* b, neuro_float& ret) override;

			bool im2col(const neuro_float* data_im, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
				neuro_float* data_col) override;

			bool col2im(const neuro_float* data_col, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
				neuro_float* data_im) override;

		private:
			_VALUE_VECTOR m_calc_buffer;
		};
	}
}
#endif
