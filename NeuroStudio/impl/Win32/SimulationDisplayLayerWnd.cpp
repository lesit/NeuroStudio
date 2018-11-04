// SimulationDisplayLayerWnd.cpp : 구현 파일입니다.
//

#include "stdafx.h"

#include "SimulationDisplayLayerWnd.h"

#include "gui/StretchAxis.h"
#include "gui/Win32/TextDraw.h"
#include "gui/Win32/ListDraw.h"
#include "gui/Win32/Win32Image.h"
#include "util/StringUtil.h"
#include "DrawLayerInfo.h"
#include "desc/LayerDesc.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// SimulationDisplayLayerWnd
using namespace np::simulate;

_LABEL_INFO layer_label(15, 5, FW_REGULAR, RGB(0, 15, 128));
_LABEL_INFO argmax_label(17, 6, FW_BOLD, RGB(80, 15, 159));
_LABEL_INFO list_label(15, 5, FW_REGULAR, RGB(0, 0, 0));

SimulationDisplayLayerWnd::SimulationDisplayLayerWnd(const _layer_display_item_matrix_vector& layer_display_item_matrix_vector)
	: m_layer_display_item_matrix_vector(layer_display_item_matrix_vector)
{
	m_max_grid_layout.SetLayout({ 0, 0, 0, 0 }, { 300, 500 }, { 10, 10 });
	m_min_item_draw_size = { 100, 50 };

	m_delegate_sample = 0;
	m_isSimulatorRunning = false;

	m_layer_label_font.CreateFontIndirect(&layer_label.logFont);
	m_argmax_label_font.CreateFontIndirect(&argmax_label.logFont);
	m_list_font.CreateFontIndirect(&list_label.logFont);
}

SimulationDisplayLayerWnd::~SimulationDisplayLayerWnd()
{
}

BEGIN_MESSAGE_MAP(SimulationDisplayLayerWnd, CScrollWnd)
	ON_WM_CREATE()
	ON_WM_PAINT()
END_MESSAGE_MAP()

int SimulationDisplayLayerWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	return 0;
}

void SimulationDisplayLayerWnd::SetDelegateSample(neuro_u32 sample) 
{
	m_delegate_sample = sample;

	Invalidate(FALSE);
}

void SimulationDisplayLayerWnd::RedrawResults()
{
//	if (!m_isSimulatorRunning)
//		return;

	Invalidate(FALSE);
}

bool SimulationDisplayLayerWnd::IsDrawTargetList(const LayerDisplayItem& item) const
{
	if (item.engine->GetLayerType() != network::_layer_type::output || ((network::OutputLayer&)item.engine->m_layer).ReadLabelForTarget())
		return false;

	return item.buffer.target.GetSize() > 0;
}

NP_SIZE SimulationDisplayLayerWnd::GetScrollTotalViewSize() const
{
	CClientDC dc(const_cast<SimulationDisplayLayerWnd*>(this));

	_layout_matrix layout_matrix;
	return CalcLayoutMatrix(dc, layout_matrix);
}

neuro_u32 SimulationDisplayLayerWnd::GetScrollMoving(bool is_horz) const
{
	return is_horz ? m_max_grid_layout.grid_size.width / 5 : m_max_grid_layout.grid_size.height / 5;
}

const neuro_u32 SimulationDisplayLayerWnd::m_draw_cell_border = 1;
const neuro_u32 SimulationDisplayLayerWnd::m_draw_border = 5;
NP_SIZE SimulationDisplayLayerWnd::CalcLayoutMatrix(CDC& dc, _layout_matrix& layout_matrix) const
{
	NP_SIZE ret(0, 0);

	NP_RECT rcLabel({ 0, 0 }, m_max_grid_layout.item_margin);

	std::vector<std::pair<neuro_u32, neuro_u32>> row_height_vector;

	layout_matrix.resize(m_layer_display_item_matrix_vector.size());
	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		_layout_row_vector& layout_row_vector = layout_matrix[level];

		neuro_u32 max_width = 0;

		const _layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		row_height_vector.resize(max(row_height_vector.size(), row_vector.size()), { 0,0 });

		layout_row_vector.resize(row_vector.size());
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			const LayerDisplayItem& item = row_vector[row];
			_LAYER_LAYOUT& layout = layout_row_vector[row];

			const tensor::TensorShape& ts = item.engine->GetOutTensorShape();

			NP_SIZE result_sz(0, 0);
			if (item.display.type == project::_layer_display_type::image)
			{
				const NP_SIZE cell_draw_size(ts.GetWidth(), ts.GetHeight());
				NP_SIZE dst_cell;
				GetDisplaySize(ts.GetChannelCount(), cell_draw_size, m_max_grid_layout.item_size
					, _MAX_CELL_SIZE(m_max_grid_layout.item_size, _stretch_type::fit_down), m_draw_cell_border, result_sz, dst_cell);
			}
			else if (item.display.type == project::_layer_display_type::list)
			{
				if (item.engine->GetLayerType() == network::_layer_type::output
					&& ((network::OutputLayer&)item.engine->m_layer).ReadLabelForTarget() || item.display.is_argmax_output)
				{
					result_sz = gui::win32::TextDraw::CalculateTextSize(dc, m_max_grid_layout.item_size, L"max no : 0");
					result_sz.height += 6;
					layout.argmax_height[0] = result_sz.height;

					if (item.buffer.target.GetSize() > 0)
					{
						layout.argmax_height[1] = layout.argmax_height[0];
						result_sz.height += layout.argmax_height[1];
					}
				}

				NP_SIZE list_sz = ListDraw::GetDrawSize(dc, true, 4, ts.GetTensorSize());
				if (IsDrawTargetList(item))
					list_sz.width += ListDraw::GetDrawSize(dc, false, 4, ts.GetTensorSize()).width;

				result_sz.width = max(result_sz.width, list_sz.width);
				result_sz.height += list_sz.height;

				result_sz.width = min(result_sz.width, m_max_grid_layout.item_size.width);
				result_sz.height = min(result_sz.height, m_max_grid_layout.item_size.height);
			}

			result_sz.height += m_draw_border;

			layout.sz.width = max(m_min_item_draw_size.width, result_sz.width);
			layout.sz.height = max(m_min_item_draw_size.height, result_sz.height);

			rcLabel.right = rcLabel.left + layout.sz.width;
			layout.label_height = gui::win32::TextDraw::CalculateTextSize(dc, rcLabel
				, str_rc::LayerDesc::GetDisplayDesc(item.engine->m_layer, item.mp), true).height;

			max_width = max(layout.sz.width, max_width);
			row_height_vector[row].first = max(row_height_vector[row].first, layout.label_height);
			row_height_vector[row].second = max(row_height_vector[row].second, layout.sz.height);
		}
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
			layout_row_vector[row].sz.width=max_width;

		ret.width += max_width;
	}

	for (neuro_u32 row = 0; row < row_height_vector.size(); row++)
	{
		ret.height += row_height_vector[row].first + row_height_vector[row].second;;
		for (neuro_u32 level = 0, level_count = layout_matrix.size(); level < level_count; level++)
		{
			_layout_row_vector& layout_row_vector = layout_matrix[level];
			if (row >= layout_row_vector.size())
				continue;

			layout_row_vector[row].label_height = row_height_vector[row].first;
			layout_row_vector[row].sz.height = row_height_vector[row].second;
		}
	}
	ret.width += 2 * m_max_grid_layout.item_margin.width * layout_matrix.size();
	ret.width += m_max_grid_layout.view_margin.left + m_max_grid_layout.view_margin.right;

	ret.height += 2 * m_max_grid_layout.item_margin.height * row_height_vector.size();
	ret.height += m_max_grid_layout.view_margin.top + m_max_grid_layout.view_margin.bottom;
	return ret;
}

void SimulationDisplayLayerWnd::OnPaint()
{
	CPaintDC paintDC(this); // device context for painting

	CRect rcClient;
	GetClientRect(&rcClient);	// 전체 영역을 얻는다.

	CMemDC memDC(paintDC, rcClient);
	CDC& dc = memDC.GetDC();
	dc.FillSolidRect(&rcClient, RGB(255, 255, 255));

	dc.SetBkMode(OPAQUE);

	HDC hdc = dc.GetSafeHdc();

	_layout_matrix layout_matrix;
	CalcLayoutMatrix(dc, layout_matrix);

	Gdiplus::Pen line_pen(Gdiplus::Color(0, 0, 0), 1.1f);
	gui::win32::GraphicUtil::CompositeLinePen(line_pen, gui::_line_arrow_type::none, gui::_line_dash_type::dash);

	NP_POINT pt;
	pt.x = m_max_grid_layout.view_margin.left+ m_max_grid_layout.item_margin.width;

	for (neuro_u32 level = 0, level_count = m_layer_display_item_matrix_vector.size(); level < level_count; level++)
	{
		pt.y = m_max_grid_layout.view_margin.top + m_max_grid_layout.item_margin.height;

		_layout_row_vector& layout_row_vector = layout_matrix[level];

		const _layer_display_item_rowl_vector& row_vector = m_layer_display_item_matrix_vector[level];
		for (neuro_u32 row = 0; row < row_vector.size(); row++)
		{
			const LayerDisplayItem& item = row_vector[row];

			_LAYER_LAYOUT& layout = layout_row_vector[row];

			NP_RECT rcLabel(ViewportToWnd(pt), NP_SIZE(layout.sz.width, layout.label_height));

			dc.SetTextColor(layer_label.color);
			CFont* oldFont = dc.SelectObject(&m_layer_label_font);
			gui::win32::TextDraw::MultiText(dc, rcLabel, str_rc::LayerDesc::GetDisplayDesc(item.engine->m_layer, item.mp)
				, gui::win32::horz_align::center);

			dc.SelectObject(oldFont);

			NP_RECT rcDraw = rcLabel;
			rcDraw.top = rcLabel.bottom + m_draw_border;
			rcDraw.bottom = rcDraw.top + layout.sz.height;

			NP_RECT rcDrawLine = rcDraw;
			rcDrawLine.InflateRect(m_draw_border, m_draw_border);
			GraphicUtil::DrawRect(hdc, rcDrawLine, line_pen);
			DrawLayerOutput(dc, rcDraw, item, layout);

			pt.y += 2 * m_max_grid_layout.item_margin.height + layout.label_height + layout.sz.height;
		}
		pt.x += 2 * m_max_grid_layout.item_margin.width + (layout_row_vector.size()>0 ? layout_row_vector[0].sz.width:0);
	}
}

inline void SimulationDisplayLayerWnd::DrawLayerOutput(CDC& dc, const NP_RECT& rcDraw, const LayerDisplayItem& item, const _LAYER_LAYOUT& layout)
{
	const _NEURO_TENSOR_DATA& output = item.buffer.output;
	if (item.buffer.output.GetSize() == 0)
		return;

	const tensor::TensorShape& ts = item.engine->GetOutTensorShape();

	if (m_delegate_sample >= output.GetBatchSize())
		m_delegate_sample = 0;

	if (item.display.type == project::_layer_display_type::image)
	{
		gui::win32::CreateImage image(ts.GetChannelCount(), false, ts.GetWidth(), ts.GetHeight());
		if (image.SetData(output.GetBatchData(m_delegate_sample), output.value_size, item.buffer.low_scale, item.buffer.up_scale))
			image.Display(dc, rcDraw, _MAX_CELL_SIZE(rcDraw.GetSize(), _stretch_type::fit_down));

	}
	else if (item.display.type == project::_layer_display_type::list)
	{
		const _TYPED_TENSOR_DATA<void*, 4>& target = item.buffer.target;

		NP_RECT rcView = rcDraw;

		// IsClassifyLossType 가 맞을지 아님 producer를 고려한 ReadLabelForTarget가 맞을지
		const bool is_output = item.engine->GetLayerType() == network::_layer_type::output;
		if (is_output && (((network::OutputLayer&)item.engine->m_layer).ReadLabelForTarget() || item.display.is_argmax_output))
		{
			neuro_32 max = max_index(output.GetBatchData(m_delegate_sample), output.value_size);

			NP_RECT rcMaxStr = rcView;
			rcMaxStr.bottom = rcMaxStr.top + layout.argmax_height[0];
			rcMaxStr.left += 10;

			dc.SetTextColor(argmax_label.color);
			CFont* oldFont = dc.SelectObject(&m_argmax_label_font);

			std::wstring max_str = L"max no : ";
			max_str += util::StringUtil::Transform<wchar_t>(max);
			gui::win32::TextDraw::SingleText(dc, rcMaxStr, max_str);
			if (target.data.buffer)
			{
				rcMaxStr.top = rcMaxStr.bottom;
				rcMaxStr.bottom = rcMaxStr.top + layout.argmax_height[1];

				if (((network::OutputLayer&)item.engine->m_layer).ReadLabelForTarget())
					max = *(neuro_u32*)target.GetBatchData(m_delegate_sample);
				else
					max_index((neuro_float*)target.GetBatchData(m_delegate_sample), target.value_size);

				std::wstring max_str = L"target : ";
				max_str += util::StringUtil::Transform<wchar_t>(max);
				gui::win32::TextDraw::SingleText(dc, rcMaxStr, max_str);
			}
			dc.SelectObject(oldFont);

			rcView.top = rcMaxStr.bottom;
		}

		dc.SetTextColor(list_label.color);
		CFont* oldFont = dc.SelectObject(&m_list_font);

		NP_SIZE sz = gui::win32::ListDraw::Draw(dc, rcView, true, 4, output.GetBatchData(m_delegate_sample), output.value_size);
		if (IsDrawTargetList(item))
		{
			rcView.left = rcView.right;
			rcView.right = rcView.left + sz.width;
			gui::win32::ListDraw::Draw(dc, rcView, false, 4, (neuro_float*)target.GetBatchData(m_delegate_sample), target.value_size);
		}

		dc.SelectObject(oldFont);
	}
}
