#include "MathGpuCore.h"

#include "cuda_platform.h"

using namespace np;
using namespace np::core;
using namespace np::core::cuda;

#define _CUBLAS_INST_CHECK \
	if (!m_cublas_handle)\
	{\
		DEBUG_OUTPUT(L"no cublas handle");\
		return false;\
	}

MathGpuCore::MathGpuCore()
: m_calc_buffer(core::math_device_type::cpu, true)
{
	m_cuda_instance = NULL;
	m_cublas_handle = NULL;
}

bool MathGpuCore::gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const neuro_u32 M, const neuro_u32 N, const neuro_u32 K,
	const neuro_float alpha, const float* A, const float* B, const neuro_float beta,
	neuro_float* C)
{
	_CUBLAS_INST_CHECK

	neuro_u32 lda = (TransA == CblasNoTrans) ? K : M;
	neuro_u32 ldb = (TransB == CblasNoTrans) ? N : K;
	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
	cublasOperation_t cuTransB = (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

	if (!CudaPlatform::CublasErrorCheck(
		cublasSgemm(m_cublas_handle, cuTransB, cuTransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N))
		)
	{
		DEBUG_OUTPUT(L"failed cublasSgemm. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	return true;
}

bool MathGpuCore::gemv(const CBLAS_TRANSPOSE TransA, const neuro_u32 M, const neuro_u32 N
	, const neuro_float alpha, const neuro_float* A, const neuro_float* x
	, const neuro_float beta, neuro_float* y)
{
	_CUBLAS_INST_CHECK

	cublasOperation_t cuTransA = (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
	if (!CudaPlatform::CublasErrorCheck(
		cublasSgemv(m_cublas_handle, cuTransA, N, M
		, &alpha, A, N, x, 1
		, &beta, y, 1))
		)
	{
		DEBUG_OUTPUT(L"failed cublasSgemm. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	return true;
}

bool MathGpuCore::axpy(const neuro_u32 N, const neuro_float alpha, const neuro_float* X, neuro_float* Y)
{
	_CUBLAS_INST_CHECK

	if (!CudaPlatform::CublasErrorCheck(
		cublasSaxpy(m_cublas_handle, N, &alpha, X, 1, Y, 1))
		)
	{ 
		DEBUG_OUTPUT(L"failed cublasSgemm. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	return true;
}

bool MathGpuCore::scale(const neuro_u32 N, const neuro_float alpha, neuro_float *X)
{
	_CUBLAS_INST_CHECK

	if (!CudaPlatform::CublasErrorCheck(
		cublasSscal(m_cublas_handle, N, &alpha, X, 1))
		)
	{
		DEBUG_OUTPUT(L"failed cublasSgemm. %s", CudaPlatform::GetErrorString().c_str());
		return false;
	}

	return true;
}

bool MathGpuCore::axpby(const neuro_u32 N
	, const neuro_float alpha, const neuro_float* X
	, const neuro_float beta, neuro_float* Y)
{
	_CUBLAS_INST_CHECK

	if (!scale(N, beta, Y) || !axpy(N, alpha, X, Y))
	{
		DEBUG_OUTPUT(L"failed");
		return false;
	}

	return true;
}

__global__ void kernel_Reduce(int N, const neuron_value* input, neuron_value* block_sum)
{
	__shared__ neuron_value cache[CudaPlatform::threadsPerBlock];

	int tid = threadIdx.x;

	neuron_value thread_sum = 0;

	CUDA_KERNEL_LOOP(i, N)
	{
		thread_sum += input[i];
	}

	cache[tid] = thread_sum;	// 각 block 내에서 각 thread의 sum을 저장한다.

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tid < i)
			cache[tid] += cache[tid + i];

		__syncthreads();
		i /= 2;
	}

	if (tid == 0)	// block의 첫번째 thread 일때 block의 sum을 저장한다.
		block_sum[blockIdx.x] = cache[0];
}

bool MathGpuCore::sum(const neuro_u32 N, const neuro_float* x, neuro_float& ret)
{
	_VALUE_VECTOR gpu_blocks_out(core::math_device_type::cuda, true);
	gpu_blocks_out.Alloc(CudaPlatform::GetCudaBlockCount(N));

	kernel_Reduce << <gpu_blocks_out.count, CudaPlatform::threadsPerBlock >> >(N, x, gpu_blocks_out.buffer);
	if (!CudaPlatform::CudaErrorCheck(cudaPeekAtLastError()))
	{
		DEBUG_OUTPUT(L"kernel_Reduce ");
		return false;
	}

	if (!m_calc_buffer.Alloc(gpu_blocks_out.count))
	{
		DEBUG_OUTPUT(L"failed alloc calc buffer");
		return false;
	}
	if (!m_calc_buffer.CopyFrom(gpu_blocks_out))
	{
		DEBUG_OUTPUT(L"failed copy calc buffer");
		return false;
	}

	ret = 0;
	for (neuro_u32 i = 0; i < m_calc_buffer.count; i++)
		ret += m_calc_buffer.buffer[i];

	return true;
}

bool MathGpuCore::asum(const neuro_u32 N, const neuro_float* x, neuro_float& ret)
{
	_CUBLAS_INST_CHECK

	if (!CudaPlatform::CublasErrorCheck(
		cublasSasum(m_cublas_handle, N, x, 1, &ret))
		)
	{
		DEBUG_OUTPUT(L"failed");
		return false;
	}
	return true;
}

bool MathGpuCore::dot(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float& ret)
{
	_CUBLAS_INST_CHECK

	if (!CudaPlatform::CublasErrorCheck(
		cublasSdot(m_cublas_handle, N, a, 1, b, 1, &ret))
		)
	{
		DEBUG_OUTPUT(L"failed");
		return false;
	}
	return true;
}

__global__ void add_scalar_kernel(const neuro_u32 N, const neuro_float alpha, neuro_float* y) 
{
	CUDA_KERNEL_LOOP(index, N)
	{
		y[index] += alpha;
	}
}

bool MathGpuCore::add_scalar(const neuro_u32 N, const neuro_float alpha, neuro_float* Y)
{
	add_scalar_kernel << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >(
		N, alpha, Y);

	return CUDA_POST_KERNEL_CHECK();
}

__global__ void powx_kernel(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y) 
{
	CUDA_KERNEL_LOOP(index, N) 
	{
		y[index] = pow(a[index], alpha);
	}
}

bool MathGpuCore::powx(const neuro_u32 N, const neuro_float* a, const neuro_float alpha, neuro_float* y)
{
	powx_kernel << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >(
		N, a, alpha, y);

	return CUDA_POST_KERNEL_CHECK();
}

__global__ void sub_kernel(const int n, const neuro_float* a,
	const neuro_float* b, neuro_float* y) 
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		y[index] = a[index] - b[index];
	}
}

bool MathGpuCore::sub(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	// NOLINT_NEXT_LINE(whitespace/operators)
	sub_kernel << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >(
		N, a, b, y);

	return CUDA_POST_KERNEL_CHECK();
}

__global__ void mul_kernel(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) 
{
	CUDA_KERNEL_LOOP(index, N)
	{
		y[index] = a[index] * b[index];
	}
}

bool MathGpuCore::mul(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	mul_kernel << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >(
		N, a, b, y);

	return CUDA_POST_KERNEL_CHECK();
}

__global__ void div_kernel(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y) 
{
	CUDA_KERNEL_LOOP(index, N) 
	{
		y[index] = a[index] / b[index];
	}
}

bool MathGpuCore::div(const neuro_u32 N, const neuro_float* a, const neuro_float* b, neuro_float* y)
{
	div_kernel << <CudaPlatform::GetCudaBlockCount(N), CudaPlatform::threadsPerBlock >> >(
		N, a, b, y);

	return CUDA_POST_KERNEL_CHECK();
}

/*	im2col and col2im come from Caffe source
*/
__global__ void im2col_gpu_kernel(const int n, const neuro_float* data_im,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	neuro_float* data_col)
{
	CUDA_KERNEL_LOOP(index, n) 
	{
		const int h_index = index / width_col;
		const int h_col = h_index % height_col;
		const int w_col = index % width_col;
		const int c_im = h_index / height_col;
		const int c_col = c_im * kernel_h * kernel_w;
		const int h_offset = h_col * stride_h - pad_h;
		const int w_offset = w_col * stride_w - pad_w;
		neuro_float* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
		const neuro_float* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;
		for (int i = 0; i < kernel_h; ++i) 
		{
			for (int j = 0; j < kernel_w; ++j) 
			{
				int h_im = h_offset + i * dilation_h;
				int w_im = w_offset + j * dilation_w;
				*data_col_ptr =
					(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
					data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}

bool MathGpuCore::im2col(const neuro_float* data_im, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
	neuro_float* data_col) 
{
	const int dkernel_h = filter_extent(kernel_h, dilation_h);
	const int dkernel_w = filter_extent(kernel_w, dilation_w);

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
	int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel << <CudaPlatform::GetCudaBlockCount(num_kernels), CudaPlatform::threadsPerBlock >> >(
		num_kernels, data_im, height, width
		, kernel_h, kernel_w, pad_t, pad_l, stride_h, stride_w, dilation_h, dilation_w, height_col
		, width_col, data_col);

	return CUDA_POST_KERNEL_CHECK();
}

__global__ void col2im_gpu_kernel(const int n, const neuro_float* data_col,
	const int height, const int width, const int channels,
	const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	const int height_col, const int width_col,
	const int dkernel_h, const int dkernel_w,
	neuro_float* data_im) 
{
	CUDA_KERNEL_LOOP(index, n)
	{
		neuro_float val = 0;
		const int w_im = index % width + pad_w;
		const int h_im = (index / width) % height + pad_h;
		const int c_im = index / (width * height);
		// compute the start and end of the output
		const int w_col_start = (w_im < dkernel_w) ? 0 : (w_im - dkernel_w) / stride_w + 1;
		const int w_col_end = min(w_im / stride_w + 1, width_col);
		const int h_col_start = (h_im < dkernel_h) ? 0 : (h_im - dkernel_h) / stride_h + 1;
		const int h_col_end = min(h_im / stride_h + 1, height_col);
		// TODO: use LCM of stride and dilation to avoid unnecessary loops
		for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
			for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
				int h_k = (h_im - h_col * stride_h);
				int w_k = (w_im - w_col * stride_w);
				if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
					h_k /= dilation_h;
					w_k /= dilation_w;
					int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
						height_col + h_col) * width_col + w_col;
					val += data_col[data_col_index];
				}
			}
		}
		data_im[index] = val;
	}
}

bool MathGpuCore::col2im(const neuro_float* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_t, const int pad_b, const int pad_l, const int pad_r,
	const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
	neuro_float* data_im) 
{
	const int dkernel_h = filter_extent(kernel_h, dilation_h);
	const int dkernel_w = filter_extent(kernel_w, dilation_w);

	int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
	int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	// NOLINT_NEXT_LINE(whitespace/operators)
	col2im_gpu_kernel << <CudaPlatform::GetCudaBlockCount(num_kernels), CudaPlatform::threadsPerBlock >> >(
		num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
		pad_t, pad_l, stride_h, stride_w, dilation_h, dilation_w,
		height_col, width_col, dkernel_h, dkernel_w, data_im);

	return CUDA_POST_KERNEL_CHECK();
}
