#pragma once

#include "../nsas/NeuroStorageAllocationSystem.h"
#include "../network/NeuralNetworkTypes.h"
#include "layers/HiddenLayerEngine.h"

#include "LayerData.h"

namespace np
{
	namespace engine
	{
		namespace layers
		{
			class HiddenLayerEngine;
		}

		using namespace layers;

		struct _LAYER_DATA_INFO
		{

		};
		class LayerWeightInit;
		class WeightStoreManager
		{
		public:
			WeightStoreManager(core::math_device_type pdtype, nsas::NeuroStorageAllocationSystem& nsas);
			virtual ~WeightStoreManager();

			virtual void RegisterLayerWeight(HiddenLayerEngine* engine)
			{
				m_layer_engine_vector.push_back(engine);
			}

			virtual bool ReadAllWeights();

			virtual bool InitAllWeights(bool is_init_weight_zero, neuro_u32 history_count);

			bool WeightsToSnapshot();
			bool SnapshotToWeights();

			virtual bool UpdateWeights();

		protected:
			bool ReadWeights(const neuro_u32 nid, neuro_u32 start, const _VALUE_VECTOR& weight_buf);
			bool WriteWeights(const neuro_u32 nid, const _VALUE_VECTOR& weight_buf);

			_VALUE_VECTOR GetAccessBuffer(const _VALUE_VECTOR& buffer);

			const core::math_device_type m_pdtype;

			nsas::NeuroStorageAllocationSystem& m_nsas;

			typedef std::vector<HiddenLayerEngine*> _layer_engine_vector;
			_layer_engine_vector m_layer_engine_vector;

		private:
			_VALUE_VECTOR m_temp_rw_buffer;

			virtual bool ManageSnapshotWeights(bool snapshot);
		};

		/*	GPU에서???
		각 layer별로 layer들의 bias_spec과 각 입력에 대한 weight_spec들을 가지고 있게 하고.
		그 관리자에서 임의의 layer의 bias와 weight들을 가져올 수 있게 하고.
		메모리가 충분하면 모두다 load 시켜놓고.
		부족하면, 최소한 두개의 layer 분의 weight을 교차로 load하게 한다.
		단, propagation에서는 현재 layer를 처리중일때 다음 layer의 weight을 thread로 load 하고
		backpropagation에서는 현재 layer를 처리중일때 이전 layer의 weight을 thread로 load 하고
		*/
	}
}
