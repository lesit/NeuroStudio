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

		/*	GPU����???
		�� layer���� layer���� bias_spec�� �� �Է¿� ���� weight_spec���� ������ �ְ� �ϰ�.
		�� �����ڿ��� ������ layer�� bias�� weight���� ������ �� �ְ� �ϰ�.
		�޸𸮰� ����ϸ� ��δ� load ���ѳ���.
		�����ϸ�, �ּ��� �ΰ��� layer ���� weight�� ������ load�ϰ� �Ѵ�.
		��, propagation������ ���� layer�� ó�����϶� ���� layer�� weight�� thread�� load �ϰ�
		backpropagation������ ���� layer�� ó�����϶� ���� layer�� weight�� thread�� load �ϰ�
		*/
	}
}
