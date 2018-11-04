#include "stdafx.h"

#include "SimDefinition.h"
//#include "../view/SimulationViewManager.h"

using namespace np;
using namespace np::project;

SimDefinition::SimDefinition(project::NeuroSystemManager& stManager)
: m_nsManager(stManager)
{
}

SimDefinition::~SimDefinition()
{
}

bool SimDefinition::IsEmpty() const
{
	return m_layer_display_info_map.size() == 0;
}

void SimDefinition::SetLayerDisplayInfo(neuro_u32 layer_uid, const _LAYER_DISPLAY_INFO& info)
{
	if (info.type == project::_layer_display_type::none)
		m_layer_display_info_map.erase(layer_uid);
	else
		m_layer_display_info_map[layer_uid] = info;
}
