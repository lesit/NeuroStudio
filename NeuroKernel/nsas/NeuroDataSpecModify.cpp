#include "stdafx.h"
#include "NeuroDataSpecModify.h"

#include "NeuroDataAccessManager.h"

using namespace np;
using namespace np::nsas;

NeuroDataSpecModify::NeuroDataSpecModify(NeuroStorageAllocationSystem& nsas, _NEURO_DATA_ALLOC_SPEC& allocSpec)
: m_nsas(nsas), m_allocSpec(allocSpec), m_pointersPerBlock(GetPointersPerBlock(nsas.GetBlockSize()))
{
}

NeuroDataSpecModify::~NeuroDataSpecModify()
{
}

bool NeuroDataSpecModify::m_is_verify = true;

bool NeuroDataSpecModify::Verify(neuro_32 new_depth)
{
	if (!m_is_verify)
		return true;

	Pointer_Table_Spec_Path pt_spec_path;
	NeuroDataSpecTreeSearch nds_tree(m_nsas, m_allocSpec);
	if (!nds_tree.CreatePath(m_allocSpec.size, pt_spec_path))
	{
		DEBUG_OUTPUT(L"failed create path");
		return false;
	}

	if (pt_spec_path.total_depth != new_depth)
	{
		DEBUG_OUTPUT(L"verify : depth[%d] is strange. it must be %d", new_depth, pt_spec_path.total_depth);
		return false;
	}

	// tree ∞À¡ı


	return true;
}
