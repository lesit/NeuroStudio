#pragma once

#include "util/UniqueIdFactory.h"
#include "storage/DeviceAdaptor.h"
#include "../nsas/NeuroStorageAllocationSystem.h"
//#include "../nsas/NeuroStorageAllocationTableSpec.h"

#include "NeuroData/model/DataProviderModel.h"

#include "InputLayer.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

/*
		인간의 뇌엔 1000억개의 뉴런이 있음. 하나의 뉴런은 최소 100개에서 최대 10000개의 연결을 가지고 있음

		1000개의 layer가 있다고 가정한다면 각 layer 당 100,000,000(1억) 개의 neuron이 있는 것이다.
		1,000개의 뉴런을 하나의 그룹으로 묶는다면 하나의 layer에 100,000 개의 그룹이 있다고 가정할수 있다.(그나마 최악의 시나리오)
		즉, 총 1억개의 그룹이 있다고 가정할 수 있다.

		1. 1000개의 layer(HiddenLayer)에 대한 할당 크기는 128b * 1000 = 약 128 kb
		2. 하나의 layer(AbstractLayer)을 표현하기 위한 할당 크기는 16bytes 정도이므로 하나의 layer에 포함된 layer들의 할당 크기는 16 * 100,000 = 1,600,000 = 약 1.6 MB 
		따라서 모든 layer의 할당 크기는 1.6mb * 1000 = 약 1.6 GB
		3. 하나의 뉴런이 10,000개의 연결을 가진다면, 하나의 layer에 1000개의 뉴런이 있으므로 하나의 연결만 가지면 되므로 4byte만 있으면 된다.
		왜냐면, 두 그룹간의 연결은 1000*1000 이므로 이미 10,000를 넘기 때문이다.
		즉, 모든 layer의 연결설정에 대한 할당 크기는 4B * 100,000 = 40KB

		최종적으로 2GB 정도면 layout 정의를 할 수 있다.
		그러면 virtual memory를 사용할 수 있지 않을까? 즉, 굳이 NAS 에 직접 writing해가면서 정의할 필요가 없을것 같다.
	*/

namespace np
{
	using namespace nsas;

	namespace network
	{
		struct _LINKED_LAYER
		{
			AbstractLayer* start;
			AbstractLayer* end;
			neuro_u32 count;
		};

		typedef std::unordered_set<InputLayer*> _input_layer_set;
		typedef std::unordered_set<OutputLayer*> _output_layer_set;
		class NeuralNetwork
		{
		public:
			NeuralNetwork();
			virtual ~NeuralNetwork();

			void New();
			bool Load(device::IODeviceFactory& device);

			bool Save(bool apply_nsas_to_layer);
			bool SaveAs(device::IODeviceFactory& device, neuro_u32 block_size = 4 * 1024, bool apply_nsas_to_layer =true);

			//void WeedHiddenOut();

			bool AddLayerInLinkedList(AbstractLayer* layer, AbstractLayer* insert_prev = NULL);
			AbstractLayer* AddLayer(np::network::_layer_type type, AbstractLayer* insert_prev = NULL);
			bool DeleteLayer(AbstractLayer* layer);
			bool MoveLayerTo(AbstractLayer* layer, AbstractLayer* insert_prev = NULL);

			bool ConnectTest(AbstractLayer* from_layer, AbstractLayer* to_layer);
			bool Connect(AbstractLayer* from_layer, AbstractLayer* to_layer, AbstractLayer* insert_prev=NULL);
			bool SideConnectTest(AbstractLayer* from_layer, AbstractLayer* to_layer);
			bool SideConnect(AbstractLayer* from_layer, AbstractLayer* to_layer);
			bool DisConnect(AbstractLayer* from_layer, AbstractLayer* to_layer);

			bool ConnectTest(NetworkBindingModel* binding, AbstractLayer* to_layer);
			bool Connect(NetworkBindingModel* binding, AbstractLayer* to_layer);
			bool DisConnect(NetworkBindingModel* binding, AbstractLayer* to_layer);

			_LINKED_LAYER GetInputLayers() const { return m_input_layers; }
			_LINKED_LAYER GetHiddenLayers() const { return m_hidden_layers; }
			void SetLinkedLayers(const _LINKED_LAYER& input_layers, const _LINKED_LAYER& hidden_layers)
			{
				m_input_layers = input_layers;
				m_hidden_layers = hidden_layers;
			}

			AbstractLayer* CreateLayerInstance(network::_layer_type type);
			void DestroyLayerInstance(AbstractLayer* layer);

			const _LEARNING_INFO& GetLearningInfo() const { return m_learning_info; }
			_LEARNING_INFO& GetLearningInfo() { return m_learning_info; }
			void SetLearningInfo(const _LEARNING_INFO& info)
			{
				memcpy(&m_learning_info, &info, sizeof(network::_LEARNING_INFO));
			}

			AbstractLayer* FindLayer(neuro_u32 uid) const
			{
				_uid_layer_map::const_iterator it = m_layer_map.find(uid);
				if (it == m_layer_map.end())
					return NULL;

				return it->second;
			}

			const _input_layer_set& GetInputLayerSet() const { return m_input_layer_set; }
			const _output_layer_set& GetOutputLayerSet() const { return m_output_layer_set; }

			const _layer_data_nid_vector& GetDeletedLayerDataNidVector() const { return m_deleted_layer_data_nid_vector; }

			nsas::NeuroStorageAllocationSystem* GetLoadNSAS(){ return m_load_nsas; }

		protected:
			virtual bool LoadCompleted() { return true; };

			// vector 대신 linked list를 사용하면 삽입/삭제가 더 편하다. 어케할지 고민해보자
			_LINKED_LAYER m_input_layers;
			_LINKED_LAYER m_hidden_layers;

			_uid_layer_map m_layer_map;
			_input_layer_set m_input_layer_set;
			_output_layer_set m_output_layer_set;

			void RemoveLayerInLinkedList(AbstractLayer* layer);
		private:
			void ClearAll();
			
			device::DeviceAdaptor* m_load_device;
			nsas::NeuroStorageAllocationSystem* m_load_nsas;

			_LEARNING_INFO m_learning_info;

			util::UniqueIdFactory m_id_factory;

			_layer_data_nid_vector m_deleted_layer_data_nid_vector;
		};
	}
}

