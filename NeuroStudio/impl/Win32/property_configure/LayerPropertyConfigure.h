#pragma once

#include "../ModelPropertyWnd.h"

#include "NeuroKernel/network/NeuralNetwork.h"

using namespace network;

namespace property
{
	class SubLayerPropertyConfigure
	{
	public:
		virtual ~SubLayerPropertyConfigure() {}

		virtual void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) = 0;
		virtual void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) {}
	};

	class LayerPropertyConfigure : public ModelPropertyConfigure
	{
	public:
		LayerPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl, NetworkViewManager& view, LastSetLayerEntryVector& last_set_entries, AbstractLayer* layer)
			: ModelPropertyConfigure(list_ctrl), m_view(view), m_last_set_entries(last_set_entries)
		{
			m_layer = layer;
			m_org_type = m_layer->GetLayerType();
		}
		virtual ~LayerPropertyConfigure() {}

		_model_property_type GetPropertyType() const override { return _model_property_type::network_layer; }
		std::wstring GetPropertyName() const override
		{
			return L"Network Layer";
		}

		void ChangeLayer(AbstractLayer* layer)
		{
			m_layer = layer;
			m_org_type = m_layer->GetLayerType();
			m_org_erased_input_vector.clear();
		}

		neuro_u32 GetModelSubType() const override {
			return (neuro_u32)m_layer->GetLayerType();
		}
		std::vector<neuro_u32> GetSubTypeVector() const override;
		const wchar_t* GetSubTypeString(neuro_u32 type) const override;
		bool SubTypeChange(AbstractBindedViewManager* view, neuro_u32 type) override;

		void CompositeProperties() override;
		void PropertyChanged(CModelGridProperty* prop, bool& reload) const override;

	private:
		SubLayerPropertyConfigure* GetConfigure() const;

		NetworkViewManager& m_view;

		LastSetLayerEntryVector& m_last_set_entries;
		AbstractLayer* m_layer;

		_layer_type m_org_type;
		_slice_input_vector m_org_erased_input_vector;
	};

	class InputLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~InputLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class FcLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~FcLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class ConvLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~ConvLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class PoolLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~PoolLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class DropoutLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~DropoutLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class BnLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~BnLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class ConcatLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~ConcatLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
	};

	class RnnLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~RnnLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};

	class OutputLayerPropertyConfigure : public SubLayerPropertyConfigure
	{
	public:
		virtual ~OutputLayerPropertyConfigure() {}

		void CompositeProperties(CMFCPropertyGridCtrl& list_ctrl, AbstractLayer* layer) override;
		void PropertyChanged(CModelGridProperty* prop, AbstractLayer* layer, bool& reload) override;
	};
}
