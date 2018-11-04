// DeeplearningDesignView.cpp : 구현 파일입니다.
//

#include "stdafx.h"
#include "DeeplearningDesignView.h"


// DeeplearningDesignView

IMPLEMENT_DYNCREATE(DeeplearningDesignView, CView)

DeeplearningDesignView::DeeplearningDesignView()
: m_preprocessorWnd(*this), m_networkWnd(*this), DeepLearningDesignViewManager({&m_preprocessorWnd}, m_networkWnd)
{

}

DeeplearningDesignView::~DeeplearningDesignView()
{
}

BEGIN_MESSAGE_MAP(DeeplearningDesignView, CView)
	ON_WM_CREATE()
	ON_WM_SIZE()
END_MESSAGE_MAP()


// DeeplearningDesignView 그리기입니다.

void DeeplearningDesignView::OnDraw(CDC* pDC)
{
	CDocument* pDoc = GetDocument();
	// TODO: 여기에 그리기 코드를 추가합니다.
}


// DeeplearningDesignView 진단입니다.

#ifdef _DEBUG
void DeeplearningDesignView::AssertValid() const
{
	CView::AssertValid();
}

#ifndef _WIN32_WCE
void DeeplearningDesignView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}
#endif
#endif //_DEBUG


// DeeplearningDesignView 메시지 처리기입니다.

#include "DesignErrorOutputPane.h"

int DeeplearningDesignView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CView::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect rcDummy(0, 0, 0, 0);

	DWORD dwTitleStyle = WS_CHILD | WS_VISIBLE | WS_BORDER | WS_CLIPSIBLINGS;
	DWORD dwDesignStyle = WS_CHILD | WS_BORDER | WS_VISIBLE | WS_HSCROLL | WS_VSCROLL;

	m_preprocessorTitle.Create(L"Preprocessor", dwTitleStyle, rcDummy, this, 270);
	if (!m_preprocessorWnd.Create(NULL, NULL, dwDesignStyle, rcDummy, this, IDC_PROVIDER_WND))
		return -1;

	m_networkTitle.Create(L"Network", dwTitleStyle, rcDummy, this, 270);
	if (!m_networkWnd.Create(NULL, NULL, dwDesignStyle, rcDummy, this, IDC_NN_DESIGN_WND))
		return -1;
	return 0;
}

void DeeplearningDesignView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	CRect rc;
	GetClientRect(&rc);

	const neuro_u32 title_height = 30;

	const neuro_u32 preprocessor_height = m_preprocessorWnd.GetScrollTotalViewSize().height;
	m_preprocessorTitle.MoveWindow(0, 0, title_height, preprocessor_height);
	m_preprocessorWnd.MoveWindow(title_height, 0, rc.Width()- title_height, preprocessor_height);

	const neuro_u32 network_height = rc.Height() - preprocessor_height;
	m_networkTitle.MoveWindow(0, preprocessor_height, title_height, network_height);
	m_networkWnd.MoveWindow(title_height, preprocessor_height, rc.Width() - title_height, network_height);
}

#include "MainFrm.h"
ModelPropertyWnd& DeeplearningDesignView::GetPropertyPane()
{
	CMainFrame* pMainFrame = (CMainFrame*)AfxGetMainWnd();
	return pMainFrame->GetModelPropertyPane();
}

DesignErrorOutputPane& DeeplearningDesignView::GetErrorOutputWnd()
{
	CMainFrame* pMainFrame = (CMainFrame*)AfxGetMainWnd();
	return pMainFrame->GetDesignErrorOutputPane();
}

