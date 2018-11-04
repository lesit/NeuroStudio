#pragma once

#include "np_types.h"

#include "storage/FileDeviceAdaptor.h"
#include "../nsas/NeuroStorageAllocationSystem.h"
#include "../engine/NetworkParameter.h"

#include "../network/NeuralNetwork.h"

#include "layers/OutputLayerEngine.h"

#include <unordered_map>

using namespace np::engine::layers;

namespace np
{
	namespace device
	{
		class DeviceAdaptor;
		class IODeviceFactory;
	}

	namespace nsas
	{
		class NeuroStorageAllocationSystem;
	}

	namespace core
	{
		namespace cuda
		{
			class CudaInstance;
		}
	}

	namespace network
	{
		class AbstractLayer;
		class InputLayer;
		class HiddenLayer;
	}

	// 이 class를 또 wrapping 하면 복잡하게 SharedDataBuffer 클래스를 만들 필요가 없다.
	namespace engine
	{
		typedef std::unordered_map<neuro_u32, layers::AbstractLayerEngine*> _uid_engine_map;
		typedef std::unordered_map<const network::AbstractLayer*, layers::AbstractLayerEngine*> _engine_map;

		class WeightStoreManager;
		class SharedDataBuffers;

		class RecvBatchSignal;

		struct _PARALLEL_INSTANCE
		{
			_PARALLEL_INSTANCE();
			~_PARALLEL_INSTANCE();

			bool IsAvailable() const;

			core::cuda::CudaInstance* cuda_instance;
		};

		struct _LAYER_SCALE_INFO
		{
			neuron_value low_scale;
			neuron_value up_scale;
		};

		class NeuralNetworkEngine
		{
		public:
			static NeuralNetworkEngine* CreateInstance(core::math_device_type pdType, _PARALLEL_INSTANCE& p_instance, network::NeuralNetwork& network);
			bool Load(network::NeuralNetwork& network);

			virtual ~NeuralNetworkEngine();

			void SetOptimizer(optimizer::OptimizeInEpoch* opt) { m_net_param.optimizer = opt; }

			HiddenLayerEngine* CreateHiddenLayerEngine(const network::HiddenLayer& layer, const _engine_map& engine_map);

			bool GetLayerOutInfo(neuro_u64 uid, _LAYER_SCALE_INFO& info) const;

			WeightStoreManager& GetWeightStoreManager(){return *m_weightStoreManager;}

			const _LEARNING_INFO& GetLearningInfo() const
			{
				return m_learning_info;
			}

			std::vector<neuro_float> GetOptimizerParameters() const
			{
				if (m_load_nsas == NULL)
					return{};

				const nsas::_OPTIMIZER_PARAMETER& opt_params = m_load_nsas->GetRootEntry().opt_params;

				std::vector<neuro_float> ret;
				ret.assign(opt_params.parameters, opt_params.parameters + opt_params.count);

				return ret;
			}

			void SetOptimizerParameters(const std::vector<neuro_float>& params)
			{
				nsas::_OPTIMIZER_PARAMETER& opt_params = m_load_nsas->GetRootEntry().opt_params;
				opt_params.count = min(_countof(opt_params.parameters),  params.size());
				for (neuro_u32 i = 0; i < opt_params.count; i++)
					opt_params.parameters[i] = params[i];
			}

			void ClearOptimizerParameters()
			{
				memset(&m_load_nsas->GetRootEntry().opt_params, 0, sizeof(_OPTIMIZER_PARAMETER));
			}

			const nsas::_LEARN_HISTORY* GetLearnHistory() const
			{
				if (m_load_nsas == NULL)
					return NULL;
				return &m_load_nsas->GetRootEntry().history; 
			}

			void SetLastLearnHistory(neuron_error loss, neuro_float accuracy)
			{
				nsas::_LEARN_HISTORY& history = m_load_nsas->GetRootEntry().history;
				history.last_loss = loss;
				history.last_accuracy = accuracy;
			}

			void SaveRootEntry()
			{
				if(m_load_nsas)
					m_load_nsas->UpdateRootInfo();
			}

			_input_engine_vector& GetInputEngineVector() { return m_input_engine_vector; }
			_hidden_engine_vector& GetHiddenEngineVector(){ return m_hidden_engine_vector; }

			_output_engine_vector& GetOutputEngineVector() { return m_output_engine_vector; }
			const _output_engine_vector& GetOutputEngineVector() const { return m_output_engine_vector;}

			const _uid_engine_map& GetUidEngineMap() const{ return m_uid_engine_map; }

			bool AllocBuffers(neuro_u32 batch_size);

			const NetworkParameter& GetNetParam() const { return m_net_param; }

		protected:
			NeuralNetworkEngine(core::math_device_type pdType, _PARALLEL_INSTANCE& p_instance
				, const network::_LEARNING_INFO& learning_info, WeightStoreManager* wsm);

		protected:
			nsas::NeuroStorageAllocationSystem* m_load_nsas;

			const network::_LEARNING_INFO& m_learning_info;

			_uid_engine_map m_uid_engine_map;

			_input_engine_vector m_input_engine_vector;
			_hidden_engine_vector m_hidden_engine_vector;
			_output_engine_vector m_output_engine_vector;

			// 모든 layer의 출력 크기와 weight 크기. 네트워크의 실행에 차지할 메모리 크기를 계산할때 사용된다.
			// 네트워크 실행에 필요한 메모리는 최소한 이 크기의 1.5배 이상은 되어야 할것이다.
			neuro_size_t m_total_layer_inner_data_size;
			neuro_size_t m_total_layer_out_size;	

			struct _BUFFER_ALLOC_INFO
			{
				neuro_u32 batch_size;
			};
			_BUFFER_ALLOC_INFO m_buffer_alloc_info;

			WeightStoreManager* m_weightStoreManager;
			NetworkParameter m_net_param;
		};
	}
}
