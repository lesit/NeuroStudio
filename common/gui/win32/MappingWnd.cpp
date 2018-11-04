#include "stdafx.h"

#include "MappingWnd.h"

using namespace np::gui::win32;

NeuroUnitDragDrop::NeuroUnitDragDrop()
{
}

bool NeuroUnitDragDrop::DragDrop(const wchar_t* cf_name, const void* buffer, neuro_size_t size)
{
	bool bRet = false;

	CSharedFile sf;
	sf.Write(&size, sizeof(neuro_size_t));
	sf.Write(buffer, size);

	HGLOBAL hMem = sf.Detach();
	if (hMem)
	{
		UINT cf = RegisterClipboardFormat(cf_name);

		COleDataSource* oleSourceObj = new COleDataSource;
		oleSourceObj->CacheGlobalData(cf, hMem);
		bRet = oleSourceObj->DoDragDrop(DROPEFFECT_LINK) != DROPEFFECT_NONE;

		// oleSourceObj를 new로 안하고 지역 변수로 했더니, 화면 밖으로 나갔다 다시 들어오면 문제 생김
		// 따라서 new로 하고 아래 InternalRelease 를 호출함
		oleSourceObj->InternalRelease();
	}

	GlobalFree(hMem);

	return bRet;
}

NPDropTarget::NPDropTarget(CMappingWnd& Wnd, const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector)
	: m_wnd(Wnd)
{
	m_cf_vector = cf_vector;
}

bool NPDropTarget::GetDragSource(COleDataObject* pDataObject, _DRAG_SOURCE& source)
{
	for (neuro_u32 i = 0; i < m_cf_vector.size(); i++)
	{
		source.cf = m_cf_vector[i].cf;

		HGLOBAL hData = pDataObject->GetGlobalData(source.cf);
		if (!hData)
			continue;

		CMemFile file((BYTE*)GlobalLock(hData), GlobalSize(hData));
		
		bool bRet = false;
		if (file.Read(&source.size, sizeof(neuro_size_t)) == sizeof(neuro_size_t))
		{
			if (source.size == m_cf_vector[i].size)
			{
				source.buffer = malloc(source.size);
				bRet = file.Read(source.buffer, source.size) == source.size;
			}
		}

		GlobalUnlock(hData);

		if (bRet)
			return true;

		free(source.buffer);
	}

	return false;
}

DROPEFFECT NPDropTarget::OnDragEnter(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point)
{
	_DRAG_SOURCE source;
	if (!GetDragSource(pDataObject, source))
		return DROPEFFECT_NONE;

	_drop_test ret = m_wnd.DropTest(source, { point.x, point.y });
	free(source.buffer);
	switch (ret)
	{
	case _drop_test::link:
		return DROPEFFECT_LINK;
	case _drop_test::move:
		return DROPEFFECT_MOVE;
	}
	return DROPEFFECT_NONE;
}

DROPEFFECT NPDropTarget::OnDragOver(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point)
{
//	m_wnd.RefreshDisplay();	// drag line을 그리기 위해서

	return OnDragEnter(pWnd, pDataObject, dwKeyState, point);
}

BOOL NPDropTarget::OnDrop(CWnd* pWnd, COleDataObject* pDataObject, DROPEFFECT dropEffect, CPoint point)
{
	_DRAG_SOURCE source;
	if (!GetDragSource(pDataObject, source))
		return __super::OnDrop(pWnd, pDataObject, dropEffect, point);

	bool ret = m_wnd.Drop(source, { point.x, point.y });
	free(source.buffer);

	return ret;
}

void NPDropTarget::OnDragLeave(CWnd* pWnd)
{
	m_wnd.DragLeave();

	COleDropTarget::OnDragLeave(pWnd);
}

CMappingWnd::CMappingWnd(const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector)
: m_dropTarget(*this, cf_vector)
{
	m_normal_line_clr = Gdiplus::Color(0, 0, 0);
	m_select_line_clr = Gdiplus::Color(0, 0, 255);
	m_cur_line_clr = Gdiplus::Color(0, 176, 255);
	m_normal_line_size = 2;
	m_hittest_line_size = 5;

	m_normal_layer_brush.CreateSolidBrush(RGB(255, 255, 255));
	m_cur_layer_brush.CreateSolidBrush(RGB(194, 193, 255));
	m_select_brush.CreateSolidBrush(RGB(122, 119, 233));
}

CMappingWnd::~CMappingWnd()
{

}

BEGIN_MESSAGE_MAP(CMappingWnd, CScrollWnd)
	ON_WM_CREATE()
	ON_WM_DESTROY()
END_MESSAGE_MAP()


int CMappingWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (__super::OnCreate(lpCreateStruct) == -1)
		return -1;

	m_dropTarget.Register(this);

	return 0;
}


void CMappingWnd::OnDestroy()
{
	m_dropTarget.Revoke();

	__super::OnDestroy();
}

LRESULT CMappingWnd::WindowProc(UINT message, WPARAM wParam, LPARAM lParam)
{
	if (message == WM_PAINT)
	{
		CPaintDC paintDC(this); // device context for painting

		CRect rcClient;
		GetClientRect(&rcClient);	// 전체 영역을 얻는다.

		CMemDC memDC(paintDC, rcClient);
		CDC& dc = memDC.GetDC();
		dc.FillSolidRect(&rcClient, RGB(255, 255, 255));
		Draw(dc, rcClient);
	}
	else if (message == WM_COMMAND)
	{
		np::studio::_menu menu = (np::studio::_menu)LOWORD(wParam);
		ProcessMenuCommand(menu);
	}
	else if (message == WM_LBUTTONDOWN || message == WM_LBUTTONDBLCLK || message == WM_LBUTTONUP
		|| message == WM_RBUTTONDOWN || message == WM_RBUTTONUP || message == WM_MOUSEMOVE || message == WM_CONTEXTMENU)
	{
		NP_POINT point(LOWORD(lParam), HIWORD(lParam));
		switch (message)
		{
		case WM_MOUSEMOVE:
			MouseMoveEvent(point);
			break;
		case WM_LBUTTONDOWN:
			MouseLClickEvent(true, point);
			break;
		case WM_LBUTTONUP:
			MouseLClickEvent(false, point);
			break;
		case WM_LBUTTONDBLCLK:
			MouseLDoubleClickEvent(point);
			break;
		case WM_RBUTTONDOWN:
			MouseRClickEvent(true, point);
			break;
		case WM_RBUTTONUP:
			MouseRClickEvent(false, point);
			break;
		case WM_CONTEXTMENU:
			{
				CPoint winpt(point.x, point.y);
				ScreenToClient(&winpt);
				point = { winpt.x, winpt.y };

				ContextMenuEvent(point);
			}
			break;
		}
	}

	return __super::WindowProc(message, wParam, lParam);
}

NP_POINT CMappingWnd::GetCurrentPoint() const
{
	if (m_hWnd == NULL)
		return NP_POINT();

	CPoint wndpt;
	::GetCursorPos(&wndpt);
	ScreenToClient(&wndpt);
	return{ wndpt.x, wndpt.y };
}

void CMappingWnd::ShowMenu(NP_POINT point, const std::vector<studio::_menu_item>& menuList)
{
	if (menuList.empty())
		return;

	CPoint pt(point.x, point.y);
	ClientToScreen(&pt);

	TRACE(L"show menu\r\n");
	CMenu menu;
	menu.CreatePopupMenu();
	for (int i = 0; i<menuList.size(); i++)
	{
		if (menuList[i].id == np::studio::_menu::split)
			menu.AppendMenu(MF_SEPARATOR);
		else
		{
			DWORD flag = MF_BYCOMMAND | MF_STRING;
			if (menuList[i].is_check)
				flag |= MF_CHECKED;
			if (menuList[i].disable)
				flag |= MF_DISABLED;

			menu.AppendMenu(flag, (UINT)menuList[i].id, GetString(menuList[i].str_id));

			//			if (menuList[i].has_check)
			//				menu.CheckMenuItem((UINT)menuList[i].id, (UINT)menuList[i].is_check ? MF_CHECKED : MF_UNCHECKED);
		}
	}
	menu.TrackPopupMenu(TPM_LEFTALIGN | TPM_RIGHTBUTTON, pt.x, pt.y, this);
}

void CMappingWnd::RectTracker(NP_POINT point, NP_2DSHAPE& rc)
{
	CRectTracker tracker;
	tracker.m_nStyle = CRectTracker::dottedLine;
	if (tracker.TrackRubberBand(this, CPoint(point.x, point.y)))
	{
		CRect ret;
		tracker.GetTrueRect(ret);
		rc = NP_2DSHAPE(ret.left, ret.top, ret.Width(), ret.Height());
	}
	else
		rc.Set(point.x, point.y, 0, 0);
}

void CMappingWnd::DrawBindingLines(HDC hdc, const _binding_link_vector& binding_link_vector, const _NEURO_BINDING_LINK* selected_link, const _NEURO_BINDING_LINK* mouseover_link)
{
	CWnd* parent = GetParent();

	for (neuro_u32 i = 0; i < binding_link_vector.size(); i++)
	{
		const _NEURO_BINDING_LINK& binding_link = binding_link_vector[i];

		_CURVE_INTEGRATED_LINE line = binding_link.line;

		CPoint pt(line.bezier.start.x, line.bezier.start.y);
		parent->ClientToScreen(&pt);
		ScreenToClient(&pt);
		line.bezier.start = { pt.x, pt.y };

		for (neuro_u32 pt_i = 0; pt_i < line.bezier.points.size(); pt_i++)
		{
			pt = { line.bezier.points[pt_i].pt.x, line.bezier.points[pt_i].pt.y };
			parent->ClientToScreen(&pt);
			ScreenToClient(&pt);

			line.bezier.points[pt_i].pt = { pt.x, pt.y };
		}
		_line_type line_type;
		if(selected_link==&binding_link)
			line_type = _line_type::select;
		else if(mouseover_link == &binding_link)
			line_type = _line_type::mouseover;
		else
			line_type = _line_type::normal;

		DrawMappingLine(hdc, line_type, line, false);
	}
}

const _NEURO_BINDING_LINK* CMappingWnd::BindingHitTest(NP_POINT point, const _binding_link_vector& binding_link_vector) const
{
	CWnd* parent = GetParent();

	CPoint pt(point.x, point.y);
	ClientToScreen(&pt);
	parent->ScreenToClient(&pt);	// binding point들은 모두 부모 좌표에 따르므로
	point = { pt.x, pt.y };

	for (neuro_u32 i = 0; i < binding_link_vector.size(); i++)
	{
		const _NEURO_BINDING_LINK& binding_link = binding_link_vector[i];
		if (LineHitTest(point, binding_link.line))
			return &binding_link;
	}
	return NULL;
}
