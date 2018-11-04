
// NeuroStudioDoc.h : CNeuroStudioDoc 클래스의 인터페이스
//


#pragma once

#include "project/NeuroStudioProject.h"

class CNeuroStudioDoc : public CDocument, public np::project::ProjectManager
{
protected: // serialization에서만 만들어집니다.
	CNeuroStudioDoc();
	virtual ~CNeuroStudioDoc();
	DECLARE_DYNCREATE(CNeuroStudioDoc)

// 특성입니다.
public:

#ifdef _DEBUG
	static CString GetTestPath(CString strName);
#endif
//	virtual std::wstring GetTempNetworkDevicePath() const;

	virtual bool RequestSaveProject();
	std::string RequestNetworkSavePath(const char* name="") const override;

// 작업입니다.
// 재정의입니다.
public:
	virtual void Serialize(CArchive& ar);

// 구현입니다.
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

	virtual BOOL OnNewDocument();
	virtual BOOL OnOpenDocument(LPCTSTR lpszPathName);
	virtual BOOL OnSaveDocument(LPCTSTR lpszPathName);
	virtual void OnCloseDocument();

// 생성된 메시지 맵 함수
protected:
	DECLARE_MESSAGE_MAP()
	afx_msg void OnFileNew();
	afx_msg void OnFileOpen();
	afx_msg void OnFileSave();
	afx_msg void OnFileSaveAs();
	afx_msg void OnNeuralNetworkReplace();

private:
	bool CreateNewProject();
	np::project::NeuroStudioProject* m_project;

	std::string RequestNewProjectfileName(std::string strRecommended = "");
public:
	virtual void SetTitle(LPCTSTR lpszTitle);
};
