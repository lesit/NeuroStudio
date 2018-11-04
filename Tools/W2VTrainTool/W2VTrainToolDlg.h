
// W2VTrainToolDlg.h : 헤더 파일
//

#pragma once

#include "fasttext.h"
#include "CreateW2VTrainDoc.h"

// CW2VTrainToolDlg 대화 상자
class CW2VTrainToolDlg : public CDialogEx, public np::nlp::recv_signal
{
// 생성입니다.
public:
	CW2VTrainToolDlg(CWnd* pParent = NULL);	// 표준 생성자입니다.
	virtual ~CW2VTrainToolDlg();

// 대화 상자 데이터입니다.
	enum { IDD = IDD_W2VTRAINTOOL };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	DECLARE_MESSAGE_MAP()
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg void OnBnClickedCheckFasttextData();
	afx_msg void OnBnClickedButtonCreateExtractText();
	afx_msg void OnBnClickedButtonTextInputPath();
	afx_msg void OnBnClickedButtonTrainModelPath();
	afx_msg void OnBnClickedButtonTrainTextFilePath();
	afx_msg void OnBnClickedButtonW2vTrain();
	afx_msg void OnBnClickedButtonExecute();
	afx_msg void OnBnClickedButtonLoad();

	afx_msg LRESULT DisplayCreateW2VDocStatus(WPARAM w, LPARAM l);

protected:
	void signal(const np::nlp::_recv_status& status) override;

	void SetTrainModulePath(const wchar_t* path);

private:
	fasttext::FastText m_fasttext;
	fasttext::Matrix* m_load_matrix;

	bool m_in_creatdoc;
public:
	afx_msg void OnBnClickedButtonTextTargetPath();
};
