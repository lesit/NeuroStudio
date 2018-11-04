#pragma once

#include "gui/line.h"

namespace np
{
	class NeuroBindingModel
	{
	public:
		virtual ~NeuroBindingModel() {}
	};

	struct _NEURO_BINDING_LINK
	{
		NeuroBindingModel* from;
		NeuroBindingModel* to;

		gui::_CURVE_INTEGRATED_LINE line;
	};
	typedef std::vector<_NEURO_BINDING_LINK> _binding_link_vector;
}
