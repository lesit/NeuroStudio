#include "stdafx.h"

#include "SharedDataBuffers.h"

using namespace np::engine;

SharedDataBuffers::SharedDataBuffers(core::math_device_type pdType)
: one_set_vector(pdType)
//, temp_gpu_layer_in_minibatch(core::math_device_type::cuda)
//, temp_gpu_layer_out_minibatch(core::math_device_type::cuda)
{
	m_max_onset_size = 0;
	m_max_onset_size_per_batch = 0;
}

SharedDataBuffers::~SharedDataBuffers()
{
	DeallocBuffers();
}

void SharedDataBuffers::DeallocBuffers()
{
	one_set_vector.Dealloc();
}

void SharedDataBuffers::SetLayerOnesetSize(neuro_u32 max_oneset_size, neuro_u32 max_oneset_size_per_batch)
{
	m_max_onset_size = max(m_max_onset_size, max_oneset_size);
	m_max_onset_size_per_batch = max(m_max_onset_size_per_batch, max_oneset_size_per_batch);
}

bool SharedDataBuffers::InitializeBuffer(neuro_u32 batch_size)
{
	DEBUG_OUTPUT(L"dealloc buffers");
	DeallocBuffers();

	DEBUG_OUTPUT(L"fixed oneset[%u], batch oneset[%u]", m_max_onset_size, m_max_onset_size_per_batch);
	neuro_u32 max_oneset_size = max(m_max_onset_size, batch_size * m_max_onset_size_per_batch);
	DEBUG_OUTPUT(L"oneset[%u] for batch[%u]", max_oneset_size, batch_size);

	// 아마도 저장할때 weight 문제가 생겼던 부분은 WeightStoreManager에서 cuda 의 문제로 수정된것 같음. 확인해보자!
	if (max_oneset_size >0 && one_set_vector.Alloc(max_oneset_size) == NULL)
	{
		DEBUG_OUTPUT(L"failed alloc one set vector");
		return false;
	}
	if (!one_set_vector.mm.DataSet(one_set_vector.buffer, 1, one_set_vector.count))
	{
		DEBUG_OUTPUT(L"failed set one vector");
		return false;
	}

	DEBUG_OUTPUT(L"end");
	return true;
}
