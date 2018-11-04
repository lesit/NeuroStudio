#pragma once

#include "shape.h"

namespace np
{
	class StretchAxis
	{
	public:
		StretchAxis()
		{
			pt.x = pt.y = 0;
			w_ratio = h_ratio = 1.0f;
		}

		StretchAxis(const NP_RECT& rc, const NP_SIZE& org_size)
		{
			pt.x = rc.left;
			pt.y = rc.top;

			NP_SIZE stretch_size = rc.GetSize();
			w_ratio = (double)stretch_size.width / (double)org_size.width;
			h_ratio = (double)stretch_size.height / (double)org_size.height;
		}

		NP_POINT Transform(const NP_POINT& src) const
		{
			return NP_POINT(pt.x + src.x*w_ratio, pt.y + src.y * h_ratio);
		}

		NP_RECT Transform(const NP_RECT& src) const
		{
			return Transform(src.left, src.top, src.right, src.bottom);
		}

		NP_RECT Transform(long left, long top, long right, long bottom) const
		{
			return NP_RECT(pt.x + left * w_ratio, pt.y + top * h_ratio
				, pt.x + right * w_ratio, pt.y + bottom * h_ratio);
		}

	private:
		NP_POINT pt;

		double w_ratio;
		double h_ratio;
	};
}
