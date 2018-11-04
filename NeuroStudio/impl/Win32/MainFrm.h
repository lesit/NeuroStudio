
// MainFrm.h : CMainFrame 클래스의 인터페이스
//

#pragma once

#include "ModelPropertyWnd.h"
#include "DesignErrorOutputPane.h"

class CMainFrame : public CFrameWndEx
{
	
protected: // serialization에서만 만들어집니다.
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// 작업입니다.
public:
	ModelPropertyWnd& GetModelPropertyPane() { return m_property_pane; }
	DesignErrorOutputPane& GetDesignErrorOutputPane() { return m_design_error_pane; }

public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // 컨트롤 모음이 포함된 멤버입니다.
	CMFCMenuBar			m_wndMenuBar;
	CMFCStatusBar		m_wndStatusBar;

	ModelPropertyWnd m_property_pane;
	DesignErrorOutputPane m_design_error_pane;

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);
	afx_msg void OnViewPropertiesWnd();
	afx_msg void OnViewErrorWnd();

	bool CreateDockingWindows();
};


