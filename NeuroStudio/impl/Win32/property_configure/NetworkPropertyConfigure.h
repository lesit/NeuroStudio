#pragma once

#include "../ModelPropertyWnd.h"

#include "NeuroKernel/network/NeuralNetwork.h"

using namespace network;

namespace property
{
	class NetworkPropertyConfigure : public ModelPropertyConfigure
	{
	public:
		NetworkPropertyConfigure(CMFCPropertyGridCtrl& list_ctrl, NeuralNetwork* network)
			: ModelPropertyConfigure(list_ctrl)
		{
			m_network = network;
		}
		virtual ~NetworkPropertyConfigure() {}

		void ChangeNetwork(NeuralNetwork* network)
		{
			m_network = network;
		}

		_model_property_type GetPropertyType() const override { return _model_property_type::neural_network; }
		std::wstring GetPropertyName() const override
		{
			return L"Neural Network";
		}

		void CompositeProperties() override;
		void PropertyChanged(CModelGridProperty* prop, bool& reload) const override;

	private:
		NeuralNetwork* m_network;
	};
}
