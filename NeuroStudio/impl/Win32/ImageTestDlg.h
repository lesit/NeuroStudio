#pragma once

#include "gui/win32/PaintWnd.h"
// CImageTestDlg 대화 상자입니다.

class CImageTestDlg : public CDialog
{
	DECLARE_DYNAMIC(CImageTestDlg)

public:
	CImageTestDlg(CWnd* pParent = NULL);   // 표준 생성자입니다.
	virtual ~CImageTestDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_IMAGE_TEST };

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
	virtual BOOL OnInitDialog();

private:
	class CImageTestWnd : public CWnd
	{
	public:
		DECLARE_MESSAGE_MAP()
		afx_msg void OnPaint();
		afx_msg BOOL OnEraseBkgnd(CDC* pDC);

		void Test1(CDC& dc, const CRect& rc);
	};
	CImageTestWnd m_testWnd;

	gui::win32::PaintCtrl m_ctrPaint;
public:
	afx_msg void OnBnClickedButtonUndo();
};
