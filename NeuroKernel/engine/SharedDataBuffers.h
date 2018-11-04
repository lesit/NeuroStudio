#if !defined(_SHARED_DATA_BUFFER_H)
#define _SHARED_DATA_BUFFER_H

#include "common.h"

namespace np
{
	namespace engine
	{
		class SharedDataBuffers
		{
		public:
			SharedDataBuffers(core::math_device_type mm_type);
			virtual ~SharedDataBuffers();

			void SetLayerOnesetSize(neuro_u32 max_oneset_size, neuro_u32 max_oneset_size_per_batch);

			bool InitializeBuffer(neuro_u32 batch_size);
			void DeallocBuffers();

			_VALUE_VECTOR one_set_vector;

		private:
			neuro_u32 m_max_onset_size;
			neuro_u32 m_max_onset_size_per_batch;
		};
	}
}

#endif
