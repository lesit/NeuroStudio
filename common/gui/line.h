#pragma once

#include "shape.h"

namespace np
{
	namespace gui
	{
		struct _BEZIER_POINT
		{
			_BEZIER_POINT() {}
			_BEZIER_POINT(const NP_POINT& pt, bool is_upward = true, neuro_float tension = 0, bool two_axis = true)
			{
				this->pt = pt;
				this->is_upward = is_upward;
				this->tension = tension;
				this->two_axis = two_axis;
			}
			NP_POINT pt;
			bool is_upward;
			neuro_float tension;
			bool two_axis;
		};
		typedef std::vector<_BEZIER_POINT> _bezier_pt_vector;

		struct _BEZIER_LINE
		{
			_BEZIER_LINE& operator = (const _BEZIER_LINE& src)
			{
				start = src.start;
				points = src.points;
				return *this;
			}

			void ViewportToWnd(const NP_POINT& vpOrg, _BEZIER_LINE& ret) const
			{
				ret.start.x = start.x - vpOrg.x;
				ret.start.y = start.y - vpOrg.y;

				ret.points.resize(points.size());
				for (neuro_u32 i = 0; i < points.size(); i++)
				{
					ret.points[i] = points[i];
					ret.points[i].pt.x -= vpOrg.x;
					ret.points[i].pt.y -= vpOrg.y;
				}
			}
			NP_POINT start;
			_bezier_pt_vector points;
		};

		enum class _line_draw_type { straight, bezier, curved };
		enum class _line_arrow_type { none, start, end, both };
		enum class _line_dash_type { solid, dot, dash, dash_dot, dash_dot_dot };
		struct _CURVE_INTEGRATED_LINE
		{
			_CURVE_INTEGRATED_LINE()
			{
				draw_type = _line_draw_type::straight;
				arrow_type = _line_arrow_type::none;
				dash_type = _line_dash_type::solid;
			}

			_CURVE_INTEGRATED_LINE& operator = (const _CURVE_INTEGRATED_LINE& src)
			{
				draw_type = src.draw_type;
				arrow_type = src.arrow_type;
				dash_type = src.dash_type;
				bezier = src.bezier;
				return *this;
			}

			_line_draw_type draw_type;
			_line_arrow_type arrow_type;
			_line_dash_type dash_type;

			_BEZIER_LINE bezier;
		};
	}
}
