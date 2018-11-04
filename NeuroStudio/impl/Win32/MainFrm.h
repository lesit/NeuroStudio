
// MainFrm.h : CMainFrame Ŭ������ �������̽�
//

#pragma once

#include "ModelPropertyWnd.h"
#include "DesignErrorOutputPane.h"

class CMainFrame : public CFrameWndEx
{
	
protected: // serialization������ ��������ϴ�.
	CMainFrame();
	DECLARE_DYNCREATE(CMainFrame)

// �۾��Դϴ�.
public:
	ModelPropertyWnd& GetModelPropertyPane() { return m_property_pane; }
	DesignErrorOutputPane& GetDesignErrorOutputPane() { return m_design_error_pane; }

public:
	virtual ~CMainFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

protected:  // ��Ʈ�� ������ ���Ե� ����Դϴ�.
	CMFCMenuBar			m_wndMenuBar;
	CMFCStatusBar		m_wndStatusBar;

	ModelPropertyWnd m_property_pane;
	DesignErrorOutputPane m_design_error_pane;

// ������ �޽��� �� �Լ�
protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnGetMinMaxInfo(MINMAXINFO* lpMMI);
	afx_msg void OnViewPropertiesWnd();
	afx_msg void OnViewErrorWnd();

	bool CreateDockingWindows();
};


