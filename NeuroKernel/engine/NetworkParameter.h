#if !defined(_NETWORK_PARAMETER_H)
#define _NETWORK_PARAMETER_H

#include "SharedDataBuffers.h"

#include "core/MathCoreApi.h"
#include "backend/optimizer.h"

namespace np
{
	namespace core
	{
		namespace cuda
		{
			class CudaInstance;
		}
	}
	namespace engine
	{
		struct NetworkParameter
		{
			NetworkParameter(const core::math_device_type _run_pdtype
							, core::cuda::CudaInstance* _cuda_instance)
				: run_pdtype(_run_pdtype)
				, math(_run_pdtype, _cuda_instance)
				, sdb(_run_pdtype)
			{
				cuda_instance = _cuda_instance;
				optimizer = NULL;
			}

			bool GetDeviceBuffer(const _NEURO_TENSOR_DATA& buffer, tensor::_ts_batch_time_order batch_time_order, _NEURO_TENSOR_DATA& using_buf) const
			{
				if (buffer.data.mm.GetType() == run_pdtype && buffer.batch_time_order == using_buf.batch_time_order)
				{
					using_buf = buffer;	// 메모리가 자동 해제되도록 하면 안된다!
				}
				else
				{
					using_buf = _NEURO_TENSOR_DATA(run_pdtype);
					using_buf.batch_time_order = batch_time_order;
					if (!using_buf.AllocLike(buffer))
					{
						DEBUG_OUTPUT(L"failed alloc");
						return false;
					}

					if (!using_buf.CopyFrom(buffer))
					{
						DEBUG_OUTPUT(L"failed copy");
						return false;
					}

					using_buf.SetAutoFree(true);
				}

				return using_buf.data.buffer != NULL;
			}

			const core::math_device_type run_pdtype;
			core::cuda::CudaInstance* cuda_instance;

			core::MathCoreApi math;

			SharedDataBuffers sdb;

			optimizer::OptimizeInEpoch* optimizer;
		};
	}
}

#endif
