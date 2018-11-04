#pragma once


// CTitleWnd

class CTitleWnd : public CWnd
{
	DECLARE_DYNAMIC(CTitleWnd)

public:
	CTitleWnd();
	virtual ~CTitleWnd();

	bool Create(const wchar_t* title, UINT dwStyle, RECT rc, CWnd* parent, int angle_degree=0);
protected:
	DECLARE_MESSAGE_MAP()
	afx_msg void OnPaint();

	CFont m_listTitleFont;

	int m_angle_degree;
public:
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
};
