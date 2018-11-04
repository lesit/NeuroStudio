// ImageTestDlg.cpp : ���� �����Դϴ�.
//

#include "stdafx.h"
#include "ImageTestDlg.h"


// CImageTestDlg ��ȭ �����Դϴ�.

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


// CImageTestDlg �޽��� ó�����Դϴ�.
BOOL CImageTestDlg::OnInitDialog()
{
	CDialog::OnInitDialog();

	// TODO:  ���⿡ �߰� �ʱ�ȭ �۾��� �߰��մϴ�.
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
	// ����: OCX �Ӽ� �������� FALSE�� ��ȯ�ؾ� �մϴ�.
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
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰� ��/�Ǵ� �⺻���� ȣ���մϴ�.

	return FALSE;
}

void CImageTestDlg::CImageTestWnd::OnPaint()
{
	CPaintDC dc(this); // device context for painting
	// TODO: ���⿡ �޽��� ó���� �ڵ带 �߰��մϴ�.
	// �׸��� �޽����� ���ؼ��� CWnd::OnPaint()��(��) ȣ������ ���ʽÿ�.

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

