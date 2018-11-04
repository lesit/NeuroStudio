// ImageTestDlg.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "ImageTestDlg.h"


// CImageTestDlg 대화 상자입니다.

IMPLEMENT_DYNAMIC(CImageTestDlg, CDialog)

CImageTestDlg::CImageTestDlg(CWnd* pParent /*=NULL*/)
: CDialog(CImageTestDlg::IDD, pParent), m_ctrPaint(200, 200)
{

}

CImageTestDlg::~CImageTestDlg()
{
}

void CImageTestDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(CImageTestDlg, CDialog)
	ON_BN_CLICKED(IDC_BUTTON_UNDO, &CImageTestDlg::OnBnClickedButtonUndo)
END_MESSAGE_MAP()


// CImageTestDlg 메시지 처리기입니다.
BOOL CImageTestDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  여기에 추가 초기화 작업을 추가합니다.
	CRect rc;
	GetClientRect(&rc);

	rc.bottom-=100;

	long width = rc.Width() / 2;
	rc.right = rc.left + width;
	m_testWnd.Create(NULL, NULL, WS_CHILD | WS_VISIBLE | WS_BORDER, rc, this, IDC_IMG_VIEW);

	rc.left = rc.right;
	rc.right += width;
	m_ctrPaint.Create(NULL, NULL, WS_CHILD | WS_VISIBLE | WS_BORDER, rc, this, IDC_IMG_VIEW + 1);
	return TRUE;  // return TRUE unless you set the focus to a control
	// 예외: OCX 속성 페이지는 FALSE를 반환해야 합니다.
}

void CImageTestDlg::OnBnClickedButtonUndo()
{
	m_ctrPaint.Undo();
}

BEGIN_MESSAGE_MAP(CImageTestDlg::CImageTestWnd, CWnd)
	ON_WM_PAINT()
	ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

BOOL CImageTestDlg::CImageTestWnd::OnEraseBkgnd(CDC* pDC)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	return FALSE;
}

void CImageTestDlg::CImageTestWnd::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
	// 그리기 메시지에 대해서는 CWnd::OnPaint()을(를) 호출하지 마십시오.

	CRect rc;
	GetClientRect(&rc);

	CMemDC memDC(dc, rc);
	memDC.GetDC().FillSolidRect(rc, RGB(255, 255, 255));

	/*
	Test1(memDC.GetDC(), rc);
	*/
}

#include "gui/Win32/Win32Image.h"

void CImageTestDlg::CImageTestWnd::Test1(CDC& dc, const CRect& rc)
{
	CString path = L"Z:\\AI & Human\\Images\\consumerpattern.jpg";

	gui::win32::ReadImage img(3, 200, 200);
	if (img.LoadImage(path))
	{
		img.GetImage().BitBlt(dc.GetSafeHdc(), 0, 0);

		_VALUE_VECTOR buf;
		buf.Alloc(img.GetDataSize());

		img.ReadData(buf.buffer, buf.count);

		gui::win32::CreateImage new_img(3, false, 200, 200, 2);
		new_img.SetData(buf.buffer, buf.count);
		new_img.Display(dc.GetSafeHdc(), NP_RECT(200, 200, rc.right, rc.bottom)
			, _MAX_CELL_SIZE(rc.right - 200, rc.bottom - 200, _stretch_type::vert_limit), true);

		buf.Dealloc();
	}
}

