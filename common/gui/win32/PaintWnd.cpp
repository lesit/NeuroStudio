#include "stdafx.h"

#include "PaintWnd.h"

using namespace np::gui;
using namespace np::gui::win32;

UINT NPM_PAINT_HAS_UNDO = ::RegisterWindowMessage(_T("NP.Util.PaintCtrl.HasUndo"));
UINT NPM_PAINT_END_DRAW = ::RegisterWindowMessage(_T("NP.Util.PaintCtrl.EndDraw"));

UINT PaintCtrl::GetHasUndoMessage()
{
	return NPM_PAINT_HAS_UNDO;
}

UINT PaintCtrl::GetEndDrawMessage()
{
	return NPM_PAINT_END_DRAW;
}

PaintCtrl::PaintCtrl(int width, int height)
{
	m_draw_count = 0;

	m_back_color = RGB(0, 0, 0);
	m_curPen.CreatePen(PS_SOLID, 20, RGB(255, 255, 255));

	NewCanvas(width, height);

	m_bLButtonDown = false;
}

PaintCtrl::~PaintCtrl()
{
}

BEGIN_MESSAGE_MAP(PaintCtrl, CWnd)
	//{{AFX_MSG_MAP(PaintCtrl)
	ON_WM_PAINT()
	ON_WM_MOUSEMOVE()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDOWN()
END_MESSAGE_MAP()

void PaintCtrl::NewCanvas(int width, int height)
{
	if (width <= 0 || height <= 0)
		return;

	if (!m_paintImg.IsNull())
	{
		BITMAP bmpinfo;
		m_lastBitmap.GetBitmap(&bmpinfo);
		if (width == bmpinfo.bmWidth && height == bmpinfo.bmHeight)
		{
			HDC dc = m_paintImg.GetDC();
			::SetBkColor(dc, m_back_color);
			CRect rect(0, 0, width, height);
			::ExtTextOut(dc, 0, 0, ETO_OPAQUE, &rect, NULL, 0, NULL);
			m_paintImg.ReleaseDC();

			m_draw_count = 0;
			m_bHasLastBitmap = false;
			if (GetSafeHwnd() != NULL)
				Invalidate(FALSE);

			CWnd* pParent = GetParent();
			if (pParent)
				pParent->SendMessage(NPM_PAINT_HAS_UNDO, 0, 0);
			return;
		}

		m_paintImg.Destroy();
		m_lastBitmap.DeleteObject();
	}

	m_draw_count = 0;

	m_paintImg.Create(width, height, 24);

	HDC dc = m_paintImg.GetDC();
	::SetBkColor(dc, m_back_color);
	CRect rect(0, 0, width, height);
	::ExtTextOut(dc, 0, 0, ETO_OPAQUE, &rect, NULL, 0, NULL);

	m_lastBitmap.Attach(::CreateCompatibleBitmap(dc, rect.Width(), rect.Height()));

	m_paintImg.ReleaseDC();

	m_bHasLastBitmap = false;
	if (GetSafeHwnd() != NULL)
		Invalidate(FALSE);

	CWnd* pParent = GetParent();
	if (pParent)
		pParent->SendMessage(NPM_PAINT_HAS_UNDO, 0, 0);
}

void PaintCtrl::EraseAll()
{
	LastPaintShot();

	if (!m_paintImg.IsNull())
	{
		HDC dc = m_paintImg.GetDC();
		::SetBkColor(dc, m_back_color);
		CRect rect(0, 0, m_paintImg.GetWidth(), m_paintImg.GetHeight());
		::ExtTextOut(dc, 0, 0, ETO_OPAQUE, &rect, NULL, 0, NULL);
		m_paintImg.ReleaseDC();

		Invalidate(FALSE);
	}
}

void PaintCtrl::Undo()
{
	if (HasCanvas())
	{
		HDC dc = m_paintImg.GetDC();

		HDC temp_dc = ::CreateCompatibleDC(dc);
		HGDIOBJ old_temp_bmp = ::SelectObject(temp_dc, m_lastBitmap.GetSafeHandle());

		BITMAP bmpinfo;
		m_lastBitmap.GetBitmap(&bmpinfo);
		::BitBlt(dc, 0, 0, bmpinfo.bmWidth, bmpinfo.bmHeight, temp_dc, 0, 0, SRCCOPY);
		::SelectObject(temp_dc, old_temp_bmp);
		::DeleteDC(temp_dc);

		m_bHasLastBitmap = false;

		m_paintImg.ReleaseDC();

		Invalidate(FALSE);

		CWnd* pParent = GetParent();
		if (pParent)
			pParent->SendMessage(NPM_PAINT_HAS_UNDO, 0, 0);
	}
}

void PaintCtrl::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	if (m_paintImg == NULL)
		return;

	m_paintImg.BitBlt(dc.GetSafeHdc(), 0, 0, SRCCOPY);
}

void PaintCtrl::LastPaintShot()
{
	HDC dc = m_paintImg.GetDC();

	BITMAP bmpinfo;
	m_lastBitmap.GetBitmap(&bmpinfo);

	HDC temp_dc = ::CreateCompatibleDC(dc);
	HGDIOBJ old_temp_bmp = ::SelectObject(temp_dc, m_lastBitmap.GetSafeHandle());
	::BitBlt(temp_dc, 0, 0, bmpinfo.bmWidth, bmpinfo.bmHeight, dc, 0, 0, SRCCOPY);
	::SelectObject(temp_dc, old_temp_bmp);
	::DeleteDC(temp_dc);

	m_paintImg.ReleaseDC();

	m_bHasLastBitmap = true;

	CWnd* pParent = GetParent();
	if (pParent)
		pParent->SendMessage(NPM_PAINT_HAS_UNDO, 0, 1);
}

void PaintCtrl::OnLButtonDown(UINT nFlags, CPoint point)
{
	m_bLButtonDown = true;
	if (HasCanvas())
	{
		HDC dc = m_paintImg.GetDC();

		HGDIOBJ old = ::SelectObject(dc, m_curPen.GetSafeHandle());

		LastPaintShot();

		::MoveToEx(dc, point.x, point.y, NULL);
		::LineTo(dc, point.x, point.y);
		++m_draw_count;

		::SelectObject(dc, old);

		m_paintImg.ReleaseDC();

		Invalidate(FALSE);
	}
	CWnd::OnLButtonDown(nFlags, point);
}

void PaintCtrl::OnMouseMove(UINT nFlags, CPoint point)
{
	if (HasCanvas())
	{
		if (m_bLButtonDown)
		{
			HDC dc = m_paintImg.GetDC();

			HGDIOBJ old = ::SelectObject(dc, m_curPen.GetSafeHandle());
			::LineTo(dc, point.x, point.y);
			++m_draw_count;
			::SelectObject(dc, old);

			m_paintImg.ReleaseDC();

			Invalidate(FALSE);
		}
	}
	CWnd::OnMouseMove(nFlags, point);
}

void PaintCtrl::OnLButtonUp(UINT nFlags, CPoint point)
{
	m_bLButtonDown = false;
	if (HasCanvas())
	{
		if (m_draw_count > 0)
		{
			CWnd* pParent = GetParent();
			if (pParent)
				pParent->SendMessage(NPM_PAINT_END_DRAW);
		}
	}

	CWnd::OnLButtonUp(nFlags, point);
}

bool PaintCtrl::ReadData(const tensor::DataShape& shape, const neuro_float scale_min, const neuro_float scale_max, bool remove_border, neuro_float* value) const
{
	if (!HasCanvas())
		return false;

	if (shape.GetChannelCount() != 1 && shape.GetChannelCount() != 3)
		return false;

	NP_RECT rc(0, 0, m_paintImg.GetWidth(), m_paintImg.GetHeight());
	if (remove_border)
	{
		neuro_u32 min_width_border = m_paintImg.GetWidth() / 10;
		neuro_u32 min_height_border = m_paintImg.GetHeight() / 10;

		rc = gui::win32::Win32Image::GetRemoveBorderRect(m_paintImg, m_back_color, min_width_border, min_height_border);
	}

	gui::win32::ReadImage read(shape.GetChannelCount()==3, shape.GetWidth(), shape.GetHeight(), scale_min, scale_max, m_back_color);
	if (!read.LoadImage(m_paintImg, rc, np::_stretch_type::fit_down))
		return false;

	return read.ReadData(value, shape.GetDimSize());	// 직접 그리는 거니까 time을 포함시키긴 애매하다.
}

PaintWnd::PaintWnd()
{

}

PaintWnd::~PaintWnd()
{

}

#define IDC_PAINT_ERASEALL	WM_USER+1
#define IDC_PAINT_UNDO		WM_USER+2
#define IDC_PAINT_CONTROL	WM_USER+3
BEGIN_MESSAGE_MAP(PaintWnd, CWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_BN_CLICKED(IDC_PAINT_ERASEALL, OnBnClickedEraseAll)
	ON_BN_CLICKED(IDC_PAINT_UNDO, OnBnClickedUndo)
	ON_REGISTERED_MESSAGE(NPM_PAINT_HAS_UNDO, OnHasUndo)
	ON_REGISTERED_MESSAGE(NPM_PAINT_END_DRAW, OnEndDrawing)
END_MESSAGE_MAP()

int PaintWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	m_ctrEraseAllBtn.Create(L"Erase all", BS_DEFPUSHBUTTON | BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER, CRect(), this, IDC_PAINT_ERASEALL);
	m_ctrUndoBtn.Create(L"Undo", BS_PUSHBUTTON | WS_VISIBLE | WS_CHILD | WS_BORDER | WS_DISABLED, CRect(), this, IDC_PAINT_UNDO);

	m_ctrPaint.Create(NULL, NULL, WS_CHILD | WS_VISIBLE | WS_BORDER, CRect(), this, IDC_PAINT_CONTROL);
	return 0;
}

void PaintWnd::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	CRect rc(5, 5, 85, 25);
	m_ctrEraseAllBtn.MoveWindow(rc);
	rc.MoveToX(rc.right + 10);
	m_ctrUndoBtn.MoveWindow(rc);

	rc.top = rc.bottom + 5;
	rc.bottom = cy - 5;
	rc.left = 5;
	rc.right = cx - 5;
	m_ctrPaint.MoveWindow(rc);
}

LRESULT PaintWnd::OnHasUndo(WPARAM wParam, LPARAM lParam)
{
	m_ctrUndoBtn.EnableWindow((BOOL)lParam);
	return 0L;
}

LRESULT PaintWnd::OnEndDrawing(WPARAM wParam, LPARAM lParam)
{
	CWnd* pParent = GetParent();
	if (pParent)
		pParent->SendMessage(NPM_PAINT_END_DRAW, wParam, lParam);
	return 0L;
}

void PaintWnd::OnBnClickedEraseAll()
{
	m_ctrPaint.EraseAll();
}

void PaintWnd::OnBnClickedUndo()
{
	m_ctrPaint.Undo();
}
