#include "stdafx.h"

#include "MathCoreApi.h"

#include "cuda_platform.h"

#include "MathCpuCore.h"
#include "MathGpuCore.h"

using namespace np;
using namespace np::core;

np::core::MathCpuCore cpu_math;
np::core::MathGpuCore gpu_math;

MathCoreApi::MathCoreApi(math_device_type _pdtype, cuda::CudaInstance* _cuda_instance)
: m_pdtype(_pdtype)
{
	if (_pdtype == math_device_type::cuda)
	{
		gpu_math.Set(_cuda_instance);
		m_core = &gpu_math;
	}
	else
		m_core = &cpu_math;
}

bool MathCoreApi::gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
	const neuro_float alpha, const neuro_float* A, const neuro_float* B, const neuro_float beta,
	neuro_float* C) const
{
	return m_core->gemm(TransA, TransB, M, N, K, alpha, A, B, beta, C);
}

bool MathCoreApi::gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
	, const neuro_float alpha, const neuro_float* A, const neuro_float* x
	, const neuro_float beta, neuro_float* y) const
{
	return m_core->gemv(TransA, M, N, alpha, A, x, beta, y);
}

bool MathCoreApi::axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y) const
{
	return m_core->axpy(N, alpha, X, Y);
}

bool MathCoreApi::scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X) const
{
	return m_core->scale(N, alpha, X);
}

bool MathCoreApi::axpby(const neuro_u32 N
	, const neuro_float alpha, const neuro_float* X
	, const neuro_float beta, neuro_float* Y) const
{
	return m_core->axpby(N, alpha, X, beta, Y);
}

bool MathCoreApi::sum(neuro_u32 N, const neuro_float* x, neuro_float& ret) const
{
	return m_core->sum(N, x, ret);
}

bool MathCoreApi::asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret) const
{
	return m_core->asum(N, x, ret);
}

bool MathCoreApi::add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y) const
{
	return m_core->add_scalar(N, alpha, Y);
}

bool MathCoreApi::powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y) const
{
	return m_core->powx(N, a, alpha, y);
}

bool MathCoreApi::sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const
{
	return m_core->sub(N, a, b, y);
}

bool MathCoreApi::mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const
{
	return m_core->mul(N, a, b, y);
}

bool MathCoreApi::div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) const
{
	return m_core->div(N, a, b, y);
}

bool MathCoreApi::dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret) const
{
	return m_core->dot(N, a, b, ret);
}

bool MathCoreApi::im2col(const neuro_float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	neuro_float* data_col) const
{
	return m_core->im2col(data_im, channels
		, height, width, kernel_h, kernel_w
		, pad_t, pad_b, pad_l, pad_r
		, stride_h, stride_w, dilation_h, dilation_w, data_col);
}

bool MathCoreApi::col2im(const neuro_float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	neuro_float* data_im) const
{
	return m_core->col2im(data_col, channels
		, height, width, kernel_h, kernel_w
		, pad_t, pad_b, pad_l, pad_r
		, stride_h, stride_w, dilation_h, dilation_w, data_im);
}
