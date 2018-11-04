#if !defined(_SIM_DISPLAY_DESC_H)
#define _SIM_DISPLAY_DESC_H

#include "Windows/GraphDataSourceAbstract.h"

namespace np
{
	namespace str_rc
	{
		class SimDisplayDesc
		{
		public:
			static const wchar_t* GetViewShapeString(windows::_graph_view::_shape_type type);
			static windows::_graph_view::_shape_type GetViewShapeType(const wchar_t* strType);
		};
	}
}

#endif
