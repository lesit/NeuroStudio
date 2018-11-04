#include "stdafx.h"

#include "DrawLayerImpl.h"

#include "gui/StretchAxis.h"
#include "gui/Win32/TextDraw.h"
#include "util/StringUtil.h"
#include "desc/LayerDesc.h"
#include "DrawLayerInfo.h"

#include "NeuroKernel/network/HiddenLayer.h"

_DRAW_LAYER_INFO draw_layer_info;

NP_SIZE DrawLayerDesign::GetDrawSize()
{
	return NP_SIZE(draw_layer_info.draw_size.width + 5 * 2, draw_layer_info.draw_size.height + 20 + 5 * 2);
}

NP_SIZE DrawLayerDesign::GetDrawShapeSize()
{
	return draw_layer_info.draw_size;
}

void DrawLayerDesign::Draw(CDC& dc, NP_RECT rc, const AbstractLayer& layer)
{
	CFont newFont;
	newFont.CreateFontIndirect(&draw_layer_info.layer_label.logFont);
	CFont* pOldFont = dc.SelectObject(&newFont);
	COLORREF prev_color = dc.SetTextColor(draw_layer_info.layer_label.color);

	dc.RoundRect(rc.left, rc.top+10, rc.right, rc.bottom, 10, 10);

	//	dc.SetBkMode(TRANSPARENT);
	dc.SetBkMode(OPAQUE);

	std::wstring desc;
	{
		std::wstring title = str_rc::LayerDesc::GetDetailName(layer);
		if (!title.empty())
		{
			desc += L" ";
			desc += title;
			desc += L" ";
		}
	}
	NP_RECT rcText = rc;
	rcText.bottom = rcText.top + 20;
	gui::win32::TextDraw::SingleText(dc, rcText, desc, gui::win32::horz_align::center);

	{
		NP_RECT rcShape;
		rcShape.left = rc.left + 5;
		rcShape.top = rcText.bottom + 5;
		rcShape.right = rcShape.left + draw_layer_info.draw_size.width;
		rcShape.bottom = rcShape.top + draw_layer_info.draw_size.height;

		DrawLayerDesign::DrawShape* shape = CreateLayerIcon(rcShape, layer);
		if (shape)
			shape->Draw(dc);

		delete shape;
	}
	dc.SetTextColor(prev_color);
	dc.SelectObject(pOldFont);
}

DrawLayerDesign::DrawShape* DrawLayerDesign::CreateLayerIcon(const NP_RECT& rcDraw, const AbstractLayer& layer)
{
	switch (layer.GetLayerType())
	{
	case network::_layer_type::fully_connected:
		return new DrawFullyShape(rcDraw, layer);
	case network::_layer_type::convolutional:
		return new DrawConvShape(rcDraw, layer);
	case network::_layer_type::pooling:
		return new DrawPoolingShape(rcDraw, layer);
	case network::_layer_type::dropout:
		return new DrawDropoutShape(rcDraw, layer);
	case network::_layer_type::rnn:
		return new DrawRnnShape(rcDraw, layer);
	case network::_layer_type::batch_norm:
		return new DrawBatchNormShape(rcDraw, layer);
	case network::_layer_type::concat:
		return new DrawConcatShape(rcDraw, layer);
	case network::_layer_type::input:
		return new DrawInputShape(rcDraw, layer);
	}
	return NULL;
}

DrawLayerDesign::DrawShape::DrawShape(const NP_RECT& rcDraw, neuro_u32 shape_id, const tensor::TensorShape& ts)
: m_shape_id(shape_id), m_ts(ts)
{
	if (!m_bmp.LoadBitmap(m_shape_id))
		return;

	BITMAP bmpinfo;
	m_bmp.GetBitmap(&bmpinfo);

	m_rcDraw = rcDraw;
	m_org_size = NP_SIZE(bmpinfo.bmWidth, bmpinfo.bmHeight);
}

void DrawLayerDesign::DrawShape::Draw(CDC& dc) const
{
	if (m_bmp.GetSafeHandle()==NULL)
		return;

	CDC bmpDC;
	bmpDC.CreateCompatibleDC(&dc);
	CBitmap* oldBmp = bmpDC.SelectObject(const_cast<CBitmap*>(&m_bmp));

	if (m_org_size.width == m_rcDraw.GetWidth() && m_org_size.height == m_rcDraw.GetHeight())
	{
		dc.BitBlt(m_rcDraw.left, m_rcDraw.top, m_rcDraw.GetWidth(), m_rcDraw.GetHeight(), &bmpDC, 0, 0, SRCCOPY);
	}
	else
	{
		dc.SetStretchBltMode(HALFTONE);
		dc.StretchBlt(m_rcDraw.left, m_rcDraw.top, m_rcDraw.GetWidth(), m_rcDraw.GetHeight(), &bmpDC, 0, 0, m_org_size.width, m_org_size.height, SRCCOPY);
	}
	bmpDC.SelectObject(oldBmp);

	DrawTensors(dc);
}

DrawLayerDesign::DrawTensorShape::DrawTensorShape(const NP_RECT& rcDraw, const tensor::TensorShape& ts)
: DrawShape(rcDraw, GetShapeID(ts), ts)
{
}

neuro_u32 DrawLayerDesign::DrawTensorShape::GetShapeID(const tensor::TensorShape& ts) const
{
	if (ts.GetDimSize() <= 1)
		return IDB_1D_1SHAPE;

	if (ts.GetHeight() <= 1 && ts.GetWidth() <= 1)
		return IDB_1DSHAPE;

	if (ts.GetChannelCount() <= 1)
		return IDB_2DSHAPE;

	return IDB_3DSHAPE;
}

void DrawLayerDesign::DrawTensorShape::DrawTensors(CDC& dc) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);

	// time length를 나중에 따로 분리해서 출력해야 한다.
	if (m_ts.time_length > 1)
	{
		std::wstring time = L"time : ";
		time += util::StringUtil::Transform<wchar_t>(m_ts.time_length);
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 0, m_org_size.width, 30), time, gui::win32::horz_align::center, true);
	}

	if (m_shape_id == IDB_1D_1SHAPE)	// 1
	{
	}
	else if( m_shape_id == IDB_1DSHAPE)// 1d
	{
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(103, 98, m_org_size.width, 118), util::StringUtil::Transform<wchar_t>(m_ts.GetChannelCount()), gui::win32::horz_align::left, true, true);
	}
	else if (m_shape_id == IDB_2DSHAPE)// 2d
	{
		NP_RECT rcText;
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(20, 175, 155, 195), util::StringUtil::Transform<wchar_t>(m_ts.GetWidth()), gui::win32::horz_align::center);
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(172, 78, m_org_size.width, 98), util::StringUtil::Transform<wchar_t>(m_ts.GetHeight()), gui::win32::horz_align::left, true, true);
	}
	else // 3d
	{
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 39, 47, 59), util::StringUtil::Transform<wchar_t>(m_ts.GetChannelCount()), gui::win32::horz_align::right, true, true);
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(53, 23, 176, 43), util::StringUtil::Transform<wchar_t>(m_ts.GetWidth()), gui::win32::horz_align::center);
		gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 105, 42, 125), util::StringUtil::Transform<wchar_t>(m_ts.GetHeight()), gui::win32::horz_align::right, true, true);
	}
}

void DrawLayerDesign::DrawTensorShape::GetOutConnPoints(bool isKenelConnect, _np_point_vector& points, NP_RECT& rcOut) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);

	switch (m_shape_id)
	{
	case IDB_1D_1SHAPE:
		points.push_back(stretch.Transform({ 130, 107 }));
		break;
	case IDB_1DSHAPE:
		points.push_back(stretch.Transform({ 88, 50 }));
		points.push_back(stretch.Transform({ 88, 169 }));
		break;
	case IDB_2DSHAPE:
	case IDB_3DSHAPE:
		if (isKenelConnect)
		{
			if (m_shape_id == IDB_2DSHAPE)
			{
				rcOut.left = 118;
				rcOut.top = 126;
			}
			else
			{
				rcOut.left = 171;
				rcOut.top = 171;
			}
			rcOut.right = rcOut.left + 20;
			rcOut.bottom = rcOut.top + 20;
			rcOut = stretch.Transform(rcOut);
			points.push_back({ rcOut.right, rcOut.top });
			points.push_back({ rcOut.right, rcOut.bottom });
		}
		else
		{
			if (m_shape_id == IDB_2DSHAPE)
			{
				points.push_back(stretch.Transform({ 155, 25 }));
				points.push_back(stretch.Transform({ 155, 159 }));
			}
			else
			{
				points.push_back(stretch.Transform({ 179, 50 }));
				points.push_back(stretch.Transform({ 206, 205 }));
			}
		}
		break;
	}
}

void DrawLayerDesign::DrawTensorShape::GetInConnPoints(_np_point_vector& points) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);
	switch (m_shape_id)
	{
	case IDB_1D_1SHAPE:
		points.push_back(stretch.Transform({ 90, 107 }));
		break;
	case IDB_1DSHAPE:
		points.push_back(stretch.Transform({ 48, 50 }));
		points.push_back(stretch.Transform({ 48, 169 }));
		break;
	case IDB_2DSHAPE:
		points.push_back(stretch.Transform({ 18, 25 }));
		points.push_back(stretch.Transform({ 18, 159 }));
		break;
	default:
		points.push_back(stretch.Transform({ 51, 50 }));
		points.push_back(stretch.Transform({ 78, 205 }));
		break;
	}
}

DrawLayerDesign::DrawInputShape::DrawInputShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawTensorShape(rcDraw, layer.GetOutTensorShape())
{
}

DrawLayerDesign::DrawHiddenShape::DrawHiddenShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id)
: DrawShape(rcDraw, shape_id, layer.GetOutTensorShape()), m_entry(((HiddenLayer&)layer).GetEntry())
{
	;
}

DrawLayerDesign::DrawFilterLayerShape::DrawFilterLayerShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id)
: DrawHiddenShape(rcDraw, layer, shape_id)
{

}

DrawLayerDesign::DrawConvShape::DrawConvShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawFilterLayerShape(rcDraw, layer, IDB_CONV_PLANE)
{
}

void DrawLayerDesign::DrawConvShape::DrawTensors(CDC& dc) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);

	gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 22, 75, 42), util::StringUtil::Transform<wchar_t>(m_ts.GetChannelCount()), gui::win32::horz_align::right, true, true);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(80, 8, 192, 28), util::StringUtil::Transform<wchar_t>(m_ts.GetWidth()), gui::win32::horz_align::center);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 84, 70, 104), util::StringUtil::Transform<wchar_t>(m_ts.GetHeight()), gui::win32::horz_align::right, true, true);

	gui::win32::TextDraw::SingleText(dc, stretch.Transform(7, 127, 56, 147), util::StringUtil::Transform<wchar_t>(m_entry.conv.filter.kernel_width), gui::win32::horz_align::center);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(76, 184, m_org_size.width, 204), util::StringUtil::Transform<wchar_t>(m_entry.conv.filter.kernel_height), gui::win32::horz_align::left, true, true);
}

void DrawLayerDesign::DrawConvShape::GetInConnPoints(_np_point_vector& points) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);
	points.push_back(stretch.Transform({ 20, 171 }));
	points.push_back(stretch.Transform({ 20, 213 }));
}

DrawLayerDesign::DrawPoolingShape::DrawPoolingShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawFilterLayerShape(rcDraw, layer, IDB_POOL_PLANE)
{
}

void DrawLayerDesign::DrawPoolingShape::DrawTensors(CDC& dc) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);

	gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 7, 72, 28), util::StringUtil::Transform<wchar_t>(m_ts.GetChannelCount()), gui::win32::horz_align::right, true, true);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(77, 5, 189, 25), util::StringUtil::Transform<wchar_t>(m_ts.GetWidth()), gui::win32::horz_align::center);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(0, 79, 70, 99), util::StringUtil::Transform<wchar_t>(m_ts.GetHeight()), gui::win32::horz_align::right, true, true);

	gui::win32::TextDraw::SingleText(dc, stretch.Transform(10, 138, 60, 158), util::StringUtil::Transform<wchar_t>(m_entry.pooling.filter.kernel_width), gui::win32::horz_align::center);
	gui::win32::TextDraw::SingleText(dc, stretch.Transform(68, 179, m_org_size.width, 199), util::StringUtil::Transform<wchar_t>(m_entry.pooling.filter.kernel_height), gui::win32::horz_align::left, true, true);
}

void DrawLayerDesign::DrawPoolingShape::GetInConnPoints(_np_point_vector& points) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);
	points.push_back(stretch.Transform({ 10, 167 }));
	points.push_back(stretch.Transform({ 10, 210 }));
}

DrawLayerDesign::DrawNormalLayerShape::DrawNormalLayerShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id)
: DrawHiddenShape(rcDraw, layer, shape_id)
{
}

void DrawLayerDesign::DrawNormalLayerShape::ConnDraw(CDC& dc) const
{
	DrawTensorShape dts(m_rcDraw, m_ts);
	dts.Draw(dc);
}

void DrawLayerDesign::DrawNormalLayerShape::GetInConnPoints(_np_point_vector& points) const
{
	DrawTensorShape dts(m_rcDraw, m_ts);
	dts.GetInConnPoints(points);
}

DrawLayerDesign::DrawFullyShape::DrawFullyShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawNormalLayerShape(rcDraw, layer, IDB_FULLY_PLANE)
{
}

void DrawLayerDesign::DrawFullyShape::DrawTensors(CDC& dc) const
{
	StretchAxis stretch(m_rcDraw, m_org_size);

	gui::win32::TextDraw::SingleText(dc, stretch.Transform(168, 95, m_org_size.width, 115), util::StringUtil::Transform<wchar_t>(m_ts.GetDimSize()), gui::win32::horz_align::left, true, true);
}

DrawLayerDesign::DrawDropoutShape::DrawDropoutShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawNormalLayerShape(rcDraw, layer, IDB_DROPOUT_PLANE)
{
}

DrawLayerDesign::DrawRnnShape::DrawRnnShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawNormalLayerShape(rcDraw, layer, IDB_RNN_PLANE)
{
}

DrawLayerDesign::DrawBatchNormShape::DrawBatchNormShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawNormalLayerShape(rcDraw, layer, IDB_BN_PLANE)
{
}

DrawLayerDesign::DrawConcatShape::DrawConcatShape(const NP_RECT& rcDraw, const AbstractLayer& layer)
: DrawNormalLayerShape(rcDraw, layer, IDB_CONCAT_PLANE)
{
}
