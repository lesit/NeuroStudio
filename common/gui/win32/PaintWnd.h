#pragma once

#include "Win32Image.h"

namespace np
{
	namespace gui
	{
		namespace win32
		{
			class PaintCtrl : public CWnd
			{
			public:
				PaintCtrl(int width=-1, int height=-1);
				virtual ~PaintCtrl();

				bool HasCanvas() const{
					return !m_paintImg.IsNull();
				}
				void NewCanvas(int width, int height);

				void EraseAll();
				void Undo();

				static UINT GetHasUndoMessage();
				static UINT GetEndDrawMessage();

				bool ReadData(const tensor::DataShape& shape, const neuro_float scale_min, const neuro_float scale_max, bool remove_border, neuro_float* value) const;

			protected:
				void LastPaintShot();

				COLORREF m_back_color;
				CPen m_curPen;

				int m_draw_count;

				CImage m_paintImg;
				CBitmap m_lastBitmap;
				bool m_bHasLastBitmap;

				bool m_bLButtonDown;
			protected:
				DECLARE_MESSAGE_MAP()
				afx_msg void OnPaint();
				afx_msg void OnMouseMove(UINT nFlags, CPoint point);
				afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
				afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
			};

			class PaintWnd : public CWnd
			{
			public:
				PaintWnd();
				virtual ~PaintWnd();

				PaintCtrl& GetPaintControl() { return m_ctrPaint; }
				const PaintCtrl& GetPaintControl() const { return m_ctrPaint; }
			protected:
				CButton m_ctrEraseAllBtn;
				CButton m_ctrUndoBtn;
				PaintCtrl m_ctrPaint;

			protected:
				DECLARE_MESSAGE_MAP()
				afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
				afx_msg void OnSize(UINT nType, int cx, int cy);
				afx_msg void OnBnClickedEraseAll();
				afx_msg void OnBnClickedUndo();
				afx_msg LRESULT OnHasUndo(WPARAM wParam, LPARAM lParam);
				afx_msg LRESULT OnEndDrawing(WPARAM wParam, LPARAM lParam);
			};
		}
	}
}
