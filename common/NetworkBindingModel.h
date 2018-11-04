#pragma once

#include <unordered_set>

#include "NeuroBindingLink.h"

namespace np
{
	class NetworkBindingModel;
	typedef std::unordered_set<NetworkBindingModel*> _neuro_binding_model_set;

	enum class _binding_model_type{data_producer, network_input_layer, network_hidden_layer, network_output_layer, layer_display};
	class NetworkBindingModel : public NeuroBindingModel
	{
	public:
		NetworkBindingModel()
		{
		}
		virtual ~NetworkBindingModel()
		{
			RemoveAllBinding();
		}

		virtual _binding_model_type GetBindingModelType() const = 0;
		virtual neuro_u32 GetUniqueID() const = 0;

		void ChangedDataShape()
		{
			_neuro_binding_model_set::iterator it = m_binding_set.begin();
			for (; it != m_binding_set.end(); it++)
				(*it)->ChangedBindingDataShape();
		}
		virtual void ChangedBindingDataShape() {}

		void AddBinding(NetworkBindingModel* binding)
		{
			if (!binding)
				return;

			m_binding_set.insert(binding);
			binding->m_binding_set.insert(this);

			ChangedBindingDataShape();
		}

		void RemoveBinding(NetworkBindingModel* target)
		{
			if (!target)
				return;

			m_binding_set.erase(target);
			target->m_binding_set.erase(this);
		}
		void RemoveAllBinding()
		{
			_neuro_binding_model_set::iterator it = m_binding_set.begin();
			for (; it != m_binding_set.end(); it++)
				(*it)->m_binding_set.erase(this);

			m_binding_set.clear();
		}
		_neuro_binding_model_set& GetBindingSet() { return m_binding_set; }
		const _neuro_binding_model_set& GetBindingSet() const { return m_binding_set; }

	private:
		_neuro_binding_model_set m_binding_set;
	};

	struct _BINDING_POINT
	{
		neuro_u32 x;
		neuro_u32 y;
	};
}
