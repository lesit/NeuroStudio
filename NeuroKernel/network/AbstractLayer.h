#pragma once

#include <unordered_set>
#include <unordered_map>
#include "NeuralNetworkTypes.h"
#include "tensor/tensor_shape.h"

#include "NetworkBindingModel.h"

namespace np
{
	namespace network
	{
		class HiddenLayer;
		typedef std::unordered_set<HiddenLayer*> _hiddenlayer_set;

		class AbstractLayer : public NetworkBindingModel
		{
		public:
			virtual ~AbstractLayer();

			neuro_u32 GetUniqueID() const override { return uid; }

			// ���� ���� �� ������ �����ϰ� �ϱ� ���� linked list�� ������Ų��.
			AbstractLayer* GetPrev() { return m_prev; }
			AbstractLayer* GetNext() { return m_next; }
			void SetPrev(AbstractLayer* layer) { m_prev = layer; }
			void SetNext(AbstractLayer* layer) { m_next = layer; }

			virtual network::_layer_type GetLayerType() const = 0;

			virtual bool AvailableConnectHiddenLayer() const { return true; }
			virtual bool AvailableConnectOutputLayer() const { return false; }

			virtual void CheckOutputTensor();

			virtual bool SetOutTensorShape(const tensor::TensorShape& ts) { return false; }
			const tensor::TensorShape& GetOutTensorShape() const {
				return m_out_ts;
			}

			const _hiddenlayer_set& GetOutputSet() const{ return m_output_set; }

			void RegisterOutput(HiddenLayer* layer);
			void ReleaseOutput(HiddenLayer* layer);

			virtual void OnRemove();

			const neuro_u32 uid;

			_BINDING_POINT gui_grid_point;
		protected:
			AbstractLayer(neuro_u32 uid);

			virtual tensor::TensorShape MakeOutTensorShape() const = 0;

			_hiddenlayer_set m_output_set;	// vector�� �����ϸ� RegisterOutput�� ������ ������� �Ѵ�!

			tensor::TensorShape m_out_ts;

			AbstractLayer* m_prev;
			AbstractLayer* m_next;
		};

		typedef std::unordered_map<neuro_u32, AbstractLayer*> _uid_layer_map;

		typedef std::vector<AbstractLayer*> _layer_vector;
		typedef std::unordered_set<AbstractLayer*> _layer_set;
	}
}

