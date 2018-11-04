#pragma once

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class CCustomDrawListWnd : public CWnd
			{
			public:
				CCustomDrawListWnd();
				virtual ~CCustomDrawListWnd();

				void RefreshScrollBars();

			protected:
				virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
				DECLARE_MESSAGE_MAP()
				afx_msg BOOL OnEraseBkgnd(CDC* pDC);
				afx_msg void OnPaint();
				afx_msg void OnSize(UINT nType, int cx, int cy);
				afx_msg void OnHScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar);
				afx_msg void OnVScroll(UINT nSBCode, UINT nTrackPos, CScrollBar* pScrollBar);
				afx_msg void OnLButtonUp(UINT nFlags, CPoint point);

			protected:
				virtual long GetHeaderHeight() const { return 0; }
				virtual void DrawHeader(CDC& dc, const CRect& rcArea) {}

				virtual neuro_size_t GetItemCount() const = 0;
				virtual neuro_u32 GetItemHeight() const = 0;
				virtual neuro_u32 GetItemWidth() const = 0;
				virtual neuro_u32 GetItemWidth(neuro_size_t item) const { return GetItemWidth(); }

				virtual void DrawItem(neuro_32 item, CDC& dc, const CRect& rcArea) = 0;
				virtual void OnItemLButtonUp(neuro_u32 item, long x, long y) {};

				neuro_u32 GetViewItemCount(neuro_u32 height) const;
				neuro_u32 GetMaxItemWidth() const;

				void RefreshScrollBar(int nType, int total_count, int view_count);

			private:
				COLORREF m_bkcolor;
				COLORREF m_line_color;
			};
		}
	}
}
