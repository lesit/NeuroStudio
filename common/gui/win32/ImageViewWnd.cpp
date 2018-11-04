#include "stdafx.h"

#include "ImageViewWnd.h"
#include "Win32Image.h"

using namespace np::gui::win32;

CImageViewWnd::CImageViewWnd(bool bCenter, bool bStretch)
{
	m_bCenter = bCenter;
	m_bStretch = bStretch;

	m_width = m_height = 0;
}

CImageViewWnd::~CImageViewWnd()
{
	m_imported_img_buffer.Dealloc();
}

BEGIN_MESSAGE_MAP(CImageViewWnd, CWnd)
	ON_WM_PAINT()
END_MESSAGE_MAP()

void CImageViewWnd::OnPaint()
{
	CPaintDC paintDC(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CWnd::OnPaint()을(를) 호출하지 마십시오.

	CRect rc;
	GetClientRect(&rc);

	CMemDC memDC(paintDC, rc);

	CDC& dc = memDC.GetDC();
	dc.FillSolidRect(rc, RGB(255, 255, 255));

	if (m_imported_img_buffer.buffer)
	{
		gui::win32::CreateImage new_img(1, 1, m_width, m_height, 0);
		new_img.SetData(m_imported_img_buffer.buffer, m_imported_img_buffer.count);
		new_img.Display(dc.GetSafeHdc(), NP_RECT(rc.left, rc.top, rc.right, rc.bottom)
			, m_bStretch ? _MAX_CELL_SIZE(rc.Width(), rc.Height(), _stretch_type::vert_limit) : _MAX_CELL_SIZE(), m_bCenter);
	}
}

neuron_value* CImageViewWnd::AllocBuffer(neuro_u32 width, neuro_u32 height)
{
	neuron_value* buffer = m_imported_img_buffer.Alloc(width*height);
	if (!buffer)
		return NULL;

	m_width = width;
	m_height = height;

	return buffer;
}
