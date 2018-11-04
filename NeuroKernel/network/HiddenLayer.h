#pragma once

#include "AbstractLayer.h"
#include "core/filter_calc.h"
#include "../nsas/NeuroEntryAccess.h"

#include "NeuralNetworkTypes.h"

namespace np
{
	namespace network
	{
		struct _SLICE_INPUT
		{
			AbstractLayer* layer;
			nsas::_SLICE_INFO slice_info;

			tensor::TensorShape GetTensor() const
			{
				return slice_info.GetTensor(layer->GetOutTensorShape());
			}
		};
		typedef std::vector<_SLICE_INPUT> _slice_input_vector;

		class HiddenLayerConfigure;
		class HiddenLayer : public AbstractLayer
		{
		public:
			static HiddenLayer* CreateInstance(np::network::_layer_type type, neuro_u32 uid);

			virtual ~HiddenLayer();

			virtual _binding_model_type GetBindingModelType() const override { return _binding_model_type::network_hidden_layer; }

			virtual _layer_type GetLayerType() const override;

			virtual bool AvailableChangeType() const { return true; }
			bool ChangeLayerType(_layer_type layer_type, const nsas::_LAYER_STRUCTURE_UNION* default_entry, _slice_input_vector* erased_input_vector=NULL);
			void ChangeEntry(const nsas::_LAYER_STRUCTURE_UNION& entry);
			void CheckChangedEntry();

			virtual void EntryValidation();

			tensor::TensorShape MakeOutTensorShape() const;
			virtual bool SetOutTensorShape(const tensor::TensorShape& ts);

			virtual bool HasActivation() const;
			virtual bool AvailableChangeActivation() const;
			virtual _activation_type GetActivation() const;

			virtual neuro_u32 AvailableInputCount() const;
			virtual bool AvailableSetSideInput(const HiddenLayer* input) const;

			virtual neuro_u32 GetLayerDataInfoVector(_layer_data_info_vector& info_vector) const;

			bool AttachStoredInfo(const nsas::_HIDDEN_LAYER_ENTRY& entry, const nsas::_input_entry_vector& input_vector, const _uid_layer_map& layer_map);
			void SetStoredNidSet(const nsas::_LAYER_DATA_NID_SET* nid_set);
			const nsas::_LAYER_DATA_NID_SET& GetStoredNidSet() const { return m_stored_sub_nid_set; }

			nsas::_LAYER_STRUCTURE_UNION& GetEntry() { return m_entry; }
			const nsas::_LAYER_STRUCTURE_UNION& GetEntry() const { return m_entry; }

			bool AvailableConnectHiddenLayer() const override;
			bool AvailableConnectOutputLayer() const override;
			bool IsConnectedOutputLayer() const;

			const _slice_input_vector& GetInputVector() const { return m_input_vector; }
			const _SLICE_INPUT* GetMainInput() const { return m_input_vector.size()>0 ? &m_input_vector[0] : NULL; }
			tensor::TensorShape GetMainInputTs() const {
				return m_input_vector.size()>0 ? m_input_vector[0].GetTensor() : tensor::TensorShape();
			}

			bool InsertInput(AbstractLayer* layer, AbstractLayer* insert_prev=NULL);
			bool BatchAppendInputs(_slice_input_vector input_vector);

			const HiddenLayer* GetSideInput() const { return m_side_input; }
			bool SetSideInput(HiddenLayer* input);

			bool DelInput(AbstractLayer* layer);
			bool ReleaseInput(AbstractLayer* layer);

			int FindInputIndex(AbstractLayer* layer) const;

			void CheckOutputTensor() override;

			void OnRemove() override;

			bool HasWeight() const;

			const _LAYER_WEIGHT_INFO* GetWeightInfo(_layer_data_type type) const
			{
				if (type == _layer_data_type::weight)
					return &m_weight_info;
				else if(type==_layer_data_type::bias)
					return &m_bias_info;
				return NULL;
			}
			void SetWeightInfo(_layer_data_type type, const _LAYER_WEIGHT_INFO& info)
			{
				if (type == _layer_data_type::weight)
					m_weight_info = info;
				else if (type == _layer_data_type::bias)
					m_bias_info = info;
			}
			
			_weight_init_type GetWeightInitType(_layer_data_type type) const;

			void SetActivation(network::_activation_type type) {
				if (!AvailableChangeActivation())
					return;
				m_activation_type = type;
			}

			void SetVirtualPosition(const nsas::_VIRTUAL_POSITION& vp)
			{
				memcpy(&m_virtual_position, &vp, sizeof(nsas::_VIRTUAL_POSITION));
			}
			const nsas::_VIRTUAL_POSITION& GetVirtualPosition() const { return m_virtual_position; }

		protected:
			HiddenLayer(neuro_u32 uid);

			bool SetLayerType(_layer_type type);

			virtual void OnInsertedInput(AbstractLayer* layer) {}

		protected:
			HiddenLayerConfigure* m_entry_configure;

			nsas::_LAYER_STRUCTURE_UNION m_entry;
			_activation_type m_activation_type;
			_LAYER_WEIGHT_INFO m_weight_info;
			_LAYER_WEIGHT_INFO m_bias_info;

			_slice_input_vector m_input_vector;
			HiddenLayer* m_side_input;

			nsas::_LAYER_DATA_NID_SET m_stored_sub_nid_set;

			nsas::_VIRTUAL_POSITION m_virtual_position;
		};

		typedef std::vector<HiddenLayer*> _hidden_layer_vector;
	}
}
