#if !defined(_MATH_CPU_CORE_H)
#define _MATH_CPU_CORE_H

#include "MathCoreApi.h"

#include "math_device.h"
#include "filter_calc.h"

#include "../3rd-party/openblas-v0.2.19-64/include/cblas.h"

namespace np
{
	namespace core
	{
		class MathCpuCore : public AbstractMathCore
		{
		public:
			bool gemm(const CBLAS_TRANSPOSE TransA,
				const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
				const float alpha, const float* A, const float* B, const float beta,
				float* C) override;

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
			bool mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) override;
			bool div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) override;

			bool dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret) override;

			bool im2col(const neuro_float* data_im, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuro_float* data_col) override;

			bool col2im(const neuro_float* data_col, const int channels,
				const int height, const int width, const int kernel_h, const int kernel_w,
				const int pad_t, const int pad_b, const int pad_l, const int pad_r,
				const int stride_h, const int stride_w,
				const int dilation_h, const int dilation_w,
				neuro_float* data_im) override;

		protected:
			bool is_a_ge_zero_and_a_lt_b(int a, int b);
		};
	}
}
#endif
