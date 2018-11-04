#pragma once

#include "../../common.h"
#include "../GraphDataSourceAbstract.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CGraphWnd : public CWnd
			{
			public:
				CGraphWnd(GraphDataSourceAbstract& graphSource);
				virtual ~CGraphWnd();

				void SetLeftVertLabel(bool bLeft){ m_bLeftVertLabel = bLeft; }

				void ChangedSource();

				int GetViewDataStart() const;
				int GetViewDataCount() const;

				void SelectCurDataPos(neuro_u64 pos);
				void EnsureDataPos(neuro_64 pos);

			protected:
				virtual void PreSubclassWindow();
				virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

				DECLARE_MESSAGE_MAP()
				afx_msg BOOL OnEraseBkgnd(CDC* pDC);
				afx_msg void OnPaint();
				afx_msg void OnSize(UINT nType, int cx, int cy);
				afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
				afx_msg void OnMouseMove(UINT nFlags, CPoint point);

			protected:
				void RefreshScrollBars();

				neuro_64 GetDataPosFromWindowPoint(int start_x, int x) const;
				CRect GetBarRect(const CRect& rcGraph, int nDataPos) const;

				virtual void DrawGraph(CDC& dc, const CRect& clientRect, const CRect& graphFullRect, const _graph_frame& graphFrame);

				void DrawYLabel(CDC& dc, const CRect& clientRect, const CRect& rcGraph, neuron_value nValueUnit, neuron_value lower_boundary, const std::vector<_graphLabel>& y_label_vector, const neuron_value* cur_pos_x=NULL);
				void DrawXLabel(CDC& dc, const CRect& rcHorzLabel, const std::vector<_graphLabel>& xLabelVector, bool draw_vert_line=false);

				void GetGraphFullRect(const CRect& clientRect, CRect& rc) const;

				int GetValuePosY(const CRect& rcGraph, neuron_value nValueUnit, neuron_value value) const;

				neuro_32 GetXValueWidth() const { return m_nXValueWidth; }
				neuro_32 GetXUnitRectWidth() const { return m_nXUnitRectWidth; }
				neuro_32 GetXLabelHeight() const { return m_nXLabelHeight; }
				neuro_32 GetYLabelWidth() const { return m_nYLabelWidth; }

				void DrawRect(CDC& dc, const CRect& rc);

				GraphDataSourceAbstract& m_graphSource;
			private:
				LOGFONT m_labelFont;
				LOGFONT m_curLabelFont;
				COLORREF m_curLabelTextColor;
				COLORREF m_curLabelBkColor;

				bool m_bLeftVertLabel;

				neuro_u32 m_nInnerGap;

				neuro_32 m_nXValueWidth;
				neuro_32 m_nXUnitRectWidth;

				neuro_32 m_nXLabelHeight;
				neuro_32 m_nYLabelWidth;

				CBrush m_graphScopeBrush;
				CBrush m_selectedBarBrush;
				CPen m_valuePen;

				COLORREF m_cur_bar_clr;
				COLORREF m_select_bar_clr;

				neuro_64 m_selected_bar_pos;
			};
		}
	}
}
