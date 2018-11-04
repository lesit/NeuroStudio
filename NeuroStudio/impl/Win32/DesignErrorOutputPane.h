#pragma once

#include "project/NeuroSystemManager.h"

class DeeplearningDesignView;
class DesignErrorOutputPane : public CDockablePane
{
public:
	DesignErrorOutputPane();
	virtual ~DesignErrorOutputPane();

	void SetDesignView(DeeplearningDesignView* design_view);

	void Clear();
	void SetErrorList(project::network_ready_error::ReadyError* error);
	
private:
	DeeplearningDesignView* m_design_view;

protected:
	DECLARE_MESSAGE_MAP()
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnLbnSelchangeErrorList();

	CListBox m_ctrErrorListBox;
};
