#include "stdafx.h"

#include "MathCpuCore.h"

using namespace np;
using namespace np::core;

bool MathCpuCore::gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
	const neuro_float alpha, const float* A, const float* B, const neuro_float beta,
	float* C)
{
	neuro_u32 lda = (TransA == CblasNoTrans) ? K : M;
	neuro_u32 ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);

	return true;
}

bool MathCpuCore::gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
	, const neuro_float alpha, const neuro_float* A, const neuro_float* x
	, const neuro_float beta, neuro_float* y)
{
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);

	return true;
}

bool MathCpuCore::axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y)
{
	cblas_saxpy(N, alpha, X, 1, Y, 1);

	return true;
}

bool MathCpuCore::scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X)
{
	cblas_sscal(N, alpha, X, 1);

	return true;
}

bool MathCpuCore::axpby(const neuro_u32 N
	, const neuro_float alpha, const neuro_float* X
	, const neuro_float beta, neuro_float* Y)
{
	cblas_saxpby(N, alpha, X, 1, beta, Y, 1);

	return true;
}

bool MathCpuCore::sum(const neuro_u32 N, const neuro_float* x, neuro_float& ret)
{
	ret = 0.f;
	for (neuro_u32 i = 0; i < N; i++)
		ret += x[i];
	return true;
}

bool MathCpuCore::asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret)
{
	ret = cblas_sasum(N, x, 1);
	return true;
}


bool MathCpuCore::dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret)
{
	ret = cblas_sdot(N, a, 1, b, 1);
	return true;
}

#include "util/cpu_parallel_for.h"

bool MathCpuCore::add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y)
{
	for_i(N, [&](neuro_u32 index)
	{
		Y[index] += alpha;
	});
	return true;
}

bool MathCpuCore::powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y)
{
	for_i(N, [&](neuro_u32 index)
	{
		y[index] = pow(a[index], alpha);
	});
	return true;
}

bool MathCpuCore::sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	for_i(N, [&](neuro_u32 index)
	{
		y[index] = a[index] - b[index];
	});
	return true;
}

bool MathCpuCore::mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	for_i(N, [&](neuro_u32 index)
	{
		y[index] = a[index] * b[index];
	});
	return true;
}

bool MathCpuCore::div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	for_i(N, [&](neuro_u32 index)
	{
		y[index] = a[index] / b[index];
	});
	return true;
}

/*	im2col and col2im come from Caffe source
*/
bool MathCpuCore::im2col(const neuro_float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	neuro_float* data_col)
{
	const int output_h = core::filter_output_length(height, kernel_h, stride_h, dilation_h, pad_t, pad_b);
	const int output_w = core::filter_output_length(width, kernel_w, stride_w, dilation_w, pad_l, pad_r);

	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size)
	{
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
		{
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
			{
				int in_h_index = -pad_t + kernel_row * dilation_h;
				for (int output_rows = output_h; output_rows; output_rows--)
				{
					if (is_a_ge_zero_and_a_lt_b(in_h_index, height))
					{
						int in_w_index = -pad_l + kernel_col * dilation_w;
						for (int output_col = output_w; output_col; output_col--)
						{
							if (is_a_ge_zero_and_a_lt_b(in_w_index, width))
								*(data_col++) = data_im[in_h_index * width + in_w_index];
							else
								*(data_col++) = 0;

							in_w_index += stride_w;
						}
					}
					else
					{
						for (int output_cols = output_w; output_cols; output_cols--)
							*(data_col++) = 0;
					}
					in_h_index += stride_h;
				}
			}
		}
	}
	return true;
}

bool MathCpuCore::col2im(const neuro_float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	neuro_float* data_im)
{
	memset(data_im, 0, sizeof(neuro_float)* height * width * channels);

	const int output_h = core::filter_output_length(height, kernel_h, stride_h, dilation_h, pad_t, pad_b);
	const int output_w = core::filter_output_length(width, kernel_w, stride_w, dilation_w, pad_l, pad_r);

	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size)
	{
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
		{
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
			{
				int in_h_index = -pad_t + kernel_row;
				for (int output_rows = output_h; output_rows; output_rows--)
				{
					if (is_a_ge_zero_and_a_lt_b(in_h_index, height))
					{
						int in_w_index = -pad_l + kernel_col;
						for (int output_col = output_w; output_col; output_col--)
						{
							if (is_a_ge_zero_and_a_lt_b(in_w_index, width))
								data_im[in_h_index * width + in_w_index] += *data_col;

							data_col++;
							in_w_index += stride_w;
						}
					}
					else
					{
						data_col += output_w;
					}

					in_h_index += stride_h;
				}
			}
		}
	}
	return true;
}

bool MathCpuCore::is_a_ge_zero_and_a_lt_b(int a, int b)
{
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
