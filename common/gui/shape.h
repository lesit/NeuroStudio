#pragma once

#include "../np_types.h"

namespace np
{
	struct NP_SIZE
	{
		NP_SIZE(){ width = height = 0; }
		NP_SIZE(long width, long height)
		{
			Set(width, height);
		}

		void Set(long width, long height)
		{
			this->width = width;
			this->height = height;
		}

		bool operator != (const NP_SIZE& other) const
		{
			return !(*this == other);
		}

		bool operator == (const NP_SIZE& other) const
		{
			return width == other.width && height == other.height;
		}

		NP_SIZE operator + (const NP_SIZE& other) const
		{
			return NP_SIZE(width + other.width, height + other.height);
		}

		NP_SIZE operator - (const NP_SIZE& other) const
		{
			return NP_SIZE(width - other.width, height - other.height);
		}

		NP_SIZE operator + (neuro_32 inc) const
		{
			return NP_SIZE(width + inc, height + inc);
		}

		NP_SIZE operator - (neuro_32 dec) const
		{
			return NP_SIZE(width - dec, height - dec);
		}

		long width;
		long height;
	};

	struct NP_POINT
	{
		NP_POINT()
		{
			x = y = 0;
		}

		NP_POINT(neuro_32 x, neuro_32 y)
		{
			Set(x, y);
		}

		void Set(neuro_32 x, neuro_32 y)
		{
			this->x = x;
			this->y = y;
		}

		bool operator == (const NP_POINT& other) const
		{
			return x == other.x && y == other.y;
		}
		
		bool operator != (const NP_POINT& other) const
		{
			return !(*this == other);
		}

		NP_POINT operator + (const NP_SIZE& sz)
		{
			return{ x + sz.width, y + sz.height };
		}

		NP_POINT operator - (const NP_SIZE& sz)
		{
			return{ x - sz.width, y - sz.height };
		}

		void operator += (const NP_SIZE& sz)
		{
			x += sz.width; y += sz.height;
		}

		void operator -= (const NP_SIZE& sz)
		{
			x -= sz.width; y -= sz.height;
		}

		long x;
		long y;
	};
	typedef std::vector<NP_POINT> _np_point_vector;

	struct NP_RECT
	{
		long left;
		long top;
		long right;
		long bottom;

		NP_RECT()
		{
			left = top = right = bottom = 0;
		}

		NP_RECT(long l, long t, long r, long b)
		{
			left = l;
			top = t;
			right = r;
			bottom = b;
		}

		NP_RECT(const NP_POINT& leftTop, const NP_POINT& rightBottom)
		{
			left = leftTop.x;
			top = leftTop.y;
			right = rightBottom.x;
			bottom = rightBottom.y;
		}

		NP_RECT(const NP_POINT& pt, const NP_SIZE& sz)
		{
			left = pt.x;
			top = pt.y;
			right = left + sz.width;
			bottom = top + sz.height;
		}

		NP_RECT(const NP_SIZE& sz)
		{
			left = 0;
			top = 0;
			right = sz.width;
			bottom = sz.height;
		}

		inline void SetHeight(long height)
		{
			bottom = top + height;
		}

		inline long GetHeight() const
		{
			return bottom - top;
		}

		inline void SetWidth(long width)
		{
			right = left + width;
		}

		inline long GetWidth() const
		{
			return right - left;
		}

		inline void SetSize(const NP_SIZE& sz)
		{
			SetWidth(sz.width);
			SetHeight(sz.height);
		}

		inline void InflateRect(neuro_32 width, neuro_32 height)
		{
			left -= width;
			right += width;
			top -= height;
			bottom += height;
		}

		inline void DeflateRect(neuro_32 width, neuro_32 height)
		{
			InflateRect(-width, -height);
		}

		inline NP_SIZE GetSize() const
		{
			return NP_SIZE(GetWidth(), GetHeight());
		}

		inline bool PtInRect(NP_POINT _pt) const
		{
			return _pt.x >= left && _pt.x <= right && _pt.y >= top && _pt.y <= bottom;
		}

		inline bool IsRectEmpty() const
		{
			return left >= right && top >= bottom;
		}
	};

	struct NP_2DSHAPE
	{
		NP_POINT pt;
		NP_SIZE sz;

		NP_2DSHAPE() {}
		NP_2DSHAPE(long width, long height)
		{
			Set(width, height);
		}
		NP_2DSHAPE(long x, long y, long width, long height)
		{
			Set(x, y, width, height);
		}

		NP_2DSHAPE(const NP_RECT& rc)
		{
			*this = rc;
		}

		NP_2DSHAPE& operator = (const NP_RECT& rc)
		{
			Set(rc.left, rc.top, rc.GetWidth(), rc.GetHeight());
			return *this;
		}

		NP_2DSHAPE(const NP_SIZE& sz)
		{
			pt.Set(0, 0);
			this->sz = sz;
		}

		NP_2DSHAPE(const NP_POINT& pt, const NP_SIZE& sz)
		{
			this->pt = pt;
			this->sz = sz;
		}

		void Set(long width, long height)
		{
			Set(0, 0, width, height);
		}

		void Set(long x, long y, long width, long height)
		{
			pt.Set(x, y);
			sz.Set(width, height);
		}

		neuro_32 Right() const
		{
			return pt.x + sz.width;
		}

		void SetRight(neuro_32 r)
		{
			sz.width = r - pt.x;
		}

		neuro_32 Bottom() const
		{
			return pt.y + sz.height;
		}

		void SetBottom(neuro_32 b)
		{
			sz.height = b - pt.y;
		}

		operator NP_RECT() const
		{
			NP_RECT rc;
			rc.left = pt.x;
			rc.top = pt.y;
			rc.SetWidth(sz.width);
			rc.SetHeight(sz.height);
			return rc;
		}

		bool PtInRect(NP_POINT _pt) const
		{
			return _pt.x >= pt.x && _pt.x <= pt.x + sz.width && _pt.y >= pt.y && _pt.y <= pt.y + sz.height;
		}

		bool IsRectEmpty() const
		{
			return sz.width == 0 && sz.height == 0;
		}
	};

	// targe view를 벗어나지 않는 범위에서 크기에 맞게 키우거나 줄임
	static NP_SIZE GetFitRatioShape(const NP_SIZE& source, const NP_SIZE& target)
	{
		NP_SIZE ret;

		const float height_img_ratio = (float)source.height / (float)source.width;
		const float height_draw_ratio = (float)target.height / (float)target.width;
		if (height_draw_ratio<height_img_ratio)	// 너비 비율이 길어짐
		{
			ret.width = static_cast<neuro_u32>(target.height / height_img_ratio);
			ret.height = target.height;
		}
		else	// 높이 비율이 길어짐
		{
			ret.width = target.width;
			ret.height = static_cast<neuro_u32>(target.width * height_img_ratio);
		}

		return ret;
	}

	enum class _stretch_type{ none, fit_shape, horz_limit, vert_limit, fit_up, fit_down };
	static const wchar_t* stretch_type_string[] ={ L"none", L"fit shape", L"horz limit", L"vert limit", L"fit up", L"fit down"};
	static const wchar_t* ToString(_stretch_type type)
	{
		if ((int)type >= _countof(stretch_type_string))
			return L"";
		return stretch_type_string[(int)type];
	}

	static NP_SIZE GetRatioShape(const NP_SIZE& source, const NP_SIZE& target, _stretch_type stretch_type)
	{
		if (source == target)
			return target;
		else if (stretch_type == _stretch_type::fit_shape)
			return GetFitRatioShape(source, target);
		
		if (stretch_type==_stretch_type::fit_up)	// 이게 쓰일일이 있을까? 소스가 짤리는데?
		{
			// 확대를 허용한 경우
			if (source.width <= target.width || source.height <= target.height)
				return GetFitRatioShape(source, target);
			
			stretch_type = _stretch_type::none;	// target이 더 작으므로 아무것도 안해도 된다.
		}
		else if (stretch_type == _stretch_type::fit_down)
		{
			// 축소를 허용한 경우
			if (source.width >= target.width || source.height >= target.height)
				return GetFitRatioShape(source, target);
			
			stretch_type = _stretch_type::none;	// target이 더 크므로 아무것도 안해도 된다.
		}

		// 아래 옵션은, 원본 축소만 하도록 함
		NP_SIZE ret;
		if (stretch_type == _stretch_type::none)
		{
			ret.width = source.width;
			ret.height = source.height;
		}
		else if (stretch_type == _stretch_type::horz_limit)// 높이는 키우진 않고 최대 타켓 너비만큼만
		{
			ret.width = source.width < target.width ? target.width : source.width;
			ret.height = long((float)ret.width * ((float)source.height / (float)source.width));
		}
		else if (stretch_type == _stretch_type::vert_limit)	// 너비는 키우진 않고 최대 타켓 높이만큼만
		{
			ret.height = source.height < target.height ? target.height : source.height;
			ret.width = long((float)ret.height * ((float)source.width / (float)source.height));
		}

		return ret;
	}

	static NP_RECT GetCenterShape(const NP_SIZE& source, const NP_2DSHAPE& target)
	{
		long left = target.pt.x;
		if (source.width <= target.sz.width)
			left += (target.sz.width - source.width) / 2;

		long top = target.pt.y;
		if (source.height <= target.sz.height)
			top += (target.sz.height - source.height) / 2;

		return NP_RECT(left, top, left+source.width, top+source.height);
	}

	struct _MAX_CELL_SIZE
	{
		_MAX_CELL_SIZE(long width = 0, long height = 0, _stretch_type fit_type = _stretch_type::none)
		{
			sz.width = width; sz.height = height;
			this->fit_type = fit_type;
		}
		_MAX_CELL_SIZE(const NP_SIZE& sz, _stretch_type fit_type)
		{
			this->sz = sz;
			this->fit_type = fit_type;
		}

		NP_SIZE sz;
		_stretch_type fit_type;
	};

	static void GetDisplaySize(neuro_u32 cell_count, const NP_SIZE& cell, const NP_SIZE& target
		, const _MAX_CELL_SIZE& max_cell_sz, neuro_u32 border, NP_SIZE& ret_target, NP_SIZE& ret_cell)
	{
		ret_target = target;
		ret_cell = cell;

		if (cell_count == 0)
			return;

		// 만약 최대 cell 크기가 정의되어 있고, 데이터의 cell 크기보다 크다면 fit 해야 한다.
		if (max_cell_sz.fit_type != _stretch_type::none
			&& ret_cell.width > max_cell_sz.sz.width || ret_cell.height > max_cell_sz.sz.height)
			ret_cell = np::GetRatioShape(ret_cell, max_cell_sz.sz, max_cell_sz.fit_type);

		neuro_u32 max_cols = (target.width- border) / (ret_cell.width + border);
		neuro_u32 max_rows = (target.height- border) / (ret_cell.height + border);

		if (cell_count < max_cols*max_rows)
		{
			neuro_u32 cols = static_cast<neuro_u32>(sqrt(neuro_float(cell_count)));
			if (cols > max_cols)
				cols = max_cols;
			neuro_u32 rows = static_cast<neuro_u32>(ceil(neuro_float(cell_count) / neuro_float(cols)));

			ret_target.width = cols * (ret_cell.width + border);
			ret_target.height = rows * (ret_cell.height + border);
		}
		ret_target.width += border;
		ret_target.height += border;
	}
}
