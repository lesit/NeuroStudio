
// NeuroStudioDoc.h : CNeuroStudioDoc Ŭ������ �������̽�
//


#pragma once

#include "project/NeuroStudioProject.h"

class CNeuroStudioDoc : public CDocument, public np::project::ProjectManager
{
protected: // serialization������ ��������ϴ�.
	CNeuroStudioDoc();
	virtual ~CNeuroStudioDoc();
	DECLARE_DYNCREATE(CNeuroStudioDoc)

// Ư���Դϴ�.
public:

#ifdef _DEBUG
	static CString GetTestPath(CString strName);
#endif
//	virtual std::wstring GetTempNetworkDevicePath() const;

	virtual bool RequestSaveProject();
	std::string RequestNetworkSavePath(const char* name="") const override;

// �۾��Դϴ�.
// �������Դϴ�.
public:
	virtual void Serialize(CArchive& ar);

// �����Դϴ�.
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

	virtual BOOL OnNewDocument();
	virtual BOOL OnOpenDocument(LPCTSTR lpszPathName);
	virtual BOOL OnSaveDocument(LPCTSTR lpszPathName);
	virtual void OnCloseDocument();

// ������ �޽��� �� �Լ�
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
