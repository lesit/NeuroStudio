#pragma once

#include "DesignPreprocessorWnd.h"
#include "DesignNetworkWnd.h"
#include "ModelPropertyWnd.h"
#include "DesignErrorOutputPane.h"

#include "TitleWnd.h"

// DeeplearningDesignView 뷰입니다.

class DeeplearningDesignView : public CView, public DeepLearningDesignViewManager
{
public:
	virtual ~DeeplearningDesignView();

	DesignPreprocessorWnd& GetProviderWnd() { return m_preprocessorWnd; }
	DesignNetworkWnd& GetNetworkWnd(){ return m_networkWnd; }

	DECLARE_DYNCREATE(DeeplearningDesignView)

	ModelPropertyWnd& GetPropertyPane();
	DesignErrorOutputPane& GetErrorOutputWnd();

protected:
	DeeplearningDesignView();           // 동적 만들기에 사용되는 protected 생성자입니다.

	void SetProject(NeuroStudioProject* project) override
	{
		__super::SetProject(project);

		if (m_project)
			GetPropertyPane().SetModelProperty(m_network_view, m_project->GetNSManager().GetNetwork());
		else
			GetPropertyPane().Clear();

		GetErrorOutputWnd().Clear();
	}

public:
	virtual void OnDraw(CDC* pDC);      // 이 뷰를 그리기 위해 재정의되었습니다.
#ifdef _DEBUG
	virtual void AssertValid() const;
#ifndef _WIN32_WCE
	virtual void Dump(CDumpContext& dc) const;
#endif
#endif

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);

private:
	CTitleWnd m_preprocessorTitle;
	DesignPreprocessorWnd m_preprocessorWnd;

	CTitleWnd m_networkTitle;
	DesignNetworkWnd m_networkWnd;
};
