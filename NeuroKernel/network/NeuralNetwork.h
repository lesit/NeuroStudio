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
		�ΰ��� ���� 1000�ﰳ�� ������ ����. �ϳ��� ������ �ּ� 100������ �ִ� 10000���� ������ ������ ����

		1000���� layer�� �ִٰ� �����Ѵٸ� �� layer �� 100,000,000(1��) ���� neuron�� �ִ� ���̴�.
		1,000���� ������ �ϳ��� �׷����� ���´ٸ� �ϳ��� layer�� 100,000 ���� �׷��� �ִٰ� �����Ҽ� �ִ�.(�׳��� �־��� �ó�����)
		��, �� 1�ﰳ�� �׷��� �ִٰ� ������ �� �ִ�.

		1. 1000���� layer(HiddenLayer)�� ���� �Ҵ� ũ��� 128b * 1000 = �� 128 kb
		2. �ϳ��� layer(AbstractLayer)�� ǥ���ϱ� ���� �Ҵ� ũ��� 16bytes �����̹Ƿ� �ϳ��� layer�� ���Ե� layer���� �Ҵ� ũ��� 16 * 100,000 = 1,600,000 = �� 1.6 MB 
		���� ��� layer�� �Ҵ� ũ��� 1.6mb * 1000 = �� 1.6 GB
		3. �ϳ��� ������ 10,000���� ������ �����ٸ�, �ϳ��� layer�� 1000���� ������ �����Ƿ� �ϳ��� ���Ḹ ������ �ǹǷ� 4byte�� ������ �ȴ�.
		�ֳĸ�, �� �׷찣�� ������ 1000*1000 �̹Ƿ� �̹� 10,000�� �ѱ� �����̴�.
		��, ��� layer�� ���ἳ���� ���� �Ҵ� ũ��� 4B * 100,000 = 40KB

		���������� 2GB ������ layout ���Ǹ� �� �� �ִ�.
		�׷��� virtual memory�� ����� �� ���� ������? ��, ���� NAS �� ���� writing�ذ��鼭 ������ �ʿ䰡 ������ ����.
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

			// vector ��� linked list�� ����ϸ� ����/������ �� ���ϴ�. �������� ����غ���
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

