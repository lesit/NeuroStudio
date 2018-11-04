
#pragma once

#include "DeeplearningDesignView.h"

class CNeuroStudioView : public CView
{
protected: // create from serialization only
	CNeuroStudioView();
	DECLARE_DYNCREATE(CNeuroStudioView)

// Attributes
public:

// Operations
public:

// Overrides
public:
	virtual void OnDraw(CDC* pDC);  // overridden to draw this view

protected:

// Implementation
public:
	virtual ~CNeuroStudioView();

protected:
	virtual void OnInitialUpdate();

// Generated message map functions
protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnSelchangingTab(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnSelchangedTab(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnBnClickedSimulation();

	CView* AddView(CRuntimeClass* pViewClass, const CString& strViewLabel);

private:
	CTabCtrl m_ctrTab;
	CButton m_ctrSimulationBtn;

	DeeplearningDesignView* m_networkView;
};

