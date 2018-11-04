#if !defined(_MATH_CORE_API_H)
#define _MATH_CORE_API_H

#include "../common.h"

#include "math_device.h"
#include "filter_calc.h"

#include "../3rd-party/openblas-v0.2.19-64/include/cblas.h"

namespace np
{
	namespace core
	{
		namespace cuda
		{
			class CudaInstance;
		}

		class AbstractMathCore
		{
		public:
			virtual bool gemm(const CBLAS_TRANSPOSE TransA,
				const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
				const neuro_float alpha, const neuro_float* A, const neuro_float* B, const neuro_float beta,
				neuro_float* C) = 0;

			virtual bool gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
				, const neuro_float alpha, const neuro_float* A, const neuro_float* x
				, const neuro_float beta, neuro_float* y) = 0;

			virtual bool axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y) = 0;

			virtual bool scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X) = 0;

			virtual bool axpby(const neuro_u32 N
				, const neuro_float alpha, const neuro_float* X
				, const neuro_float beta, neuro_float* Y) = 0;

			virtual bool sum(const neuro_u32 N, const neuro_float* x, neuro_float& ret)  = 0;
			virtual bool asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) = 0;

			virtual bool add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y) = 0;
			virtual bool powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y) = 0;
			virtual bool sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) = 0;
			virtual bool mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) = 0;
			virtual bool div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) = 0;

			virtual bool dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret) = 0;

			virtual bool im2col(const neuron_value* data_im, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuron_value* data_col) = 0;

			virtual bool col2im(const neuron_value* data_col, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuron_value* data_im) = 0;
		};

		class MathCoreApi
		{
		public:
			MathCoreApi(math_device_type _pdtype, cuda::CudaInstance* _cuda_instance = NULL);
			const math_device_type m_pdtype;

		public:
			bool gemm(const CBLAS_TRANSPOSE TransA,
				const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
				const neuro_float alpha, const neuro_float* A, const neuro_float* B, const neuro_float beta,
				neuro_float* C) const;

			bool gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
				, const neuro_float alpha, const neuro_float* A, const neuro_float* x
				, const neuro_float beta, neuro_float* y) const;

			bool axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y) const;

			bool scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X) const;

			bool axpby(const neuro_u32 N
				, const neuro_float alpha, const neuro_float* X
				, const neuro_float beta, neuro_float* Y) const;

			bool sum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) const;
			bool asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) const;

			bool add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y) const;
			bool powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y) const;
			bool sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const;
			bool mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const;
			bool div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const;

			bool dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret) const;

			bool im2col(const neuro_float* data_im, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuro_float* data_col) const;

			bool col2im(const neuro_float* data_col, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuro_float* data_im) const;

			AbstractMathCore* m_core;
		};
	}
}
#endif
