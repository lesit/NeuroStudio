#pragma once

#include "gui/win32/ScrollWnd.h"
#include "gui/Win32/GraphicUtil.h"
#include "StudioMenu.h"

#include "../../NeuroBindingLink.h"
#include "../line.h"

namespace np
{
	namespace gui
	{

		namespace win32
		{
			enum class _drop_test { none, link, move };

			struct _DRAG_SOURCE
			{
				neuro_u32 cf;

				void* buffer;
				neuro_size_t size;
			};

			class NeuroUnitDragDrop
			{
			public:
				NeuroUnitDragDrop();

				bool DragDrop(const wchar_t* cf_name, const void* buffer, neuro_size_t size);
			};

			struct _CLIPBOARDFORMAT_INFO
			{
				neuro_u32 cf;
				neuro_size_t size;
			};

			class CMappingWnd;
			class NPDropTarget : public COleDropTarget
			{
			public:
				NPDropTarget(CMappingWnd& Wnd, const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector = {});
				virtual ~NPDropTarget() {};

			protected:
				virtual DROPEFFECT OnDragEnter(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point);
				virtual DROPEFFECT OnDragOver(CWnd* pWnd, COleDataObject* pDataObject, DWORD dwKeyState, CPoint point);
				virtual BOOL OnDrop(CWnd* pWnd, COleDataObject* pDataObject, DROPEFFECT dropEffect, CPoint point);
				virtual void OnDragLeave(CWnd* pWnd);

				bool GetDragSource(COleDataObject* pDataObject, _DRAG_SOURCE& source);

				std::vector<_CLIPBOARDFORMAT_INFO> m_cf_vector;

				CMappingWnd& m_wnd;
			};

			class CMappingWnd : public CScrollWnd
			{
			public:
				CMappingWnd(const std::vector<_CLIPBOARDFORMAT_INFO>& cf_vector = {});
				virtual ~CMappingWnd();

				NP_POINT GetCurrentPoint() const;

				virtual _drop_test DropTest(const _DRAG_SOURCE& source, NP_POINT point) { return _drop_test::none; }
				virtual bool Drop(const _DRAG_SOURCE& source, NP_POINT point) { return false; }
				virtual void DragLeave() {}

			protected:
				DECLARE_MESSAGE_MAP()
				afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
				afx_msg void OnDestroy();
				virtual LRESULT WindowProc(UINT message, WPARAM wParam, LPARAM lParam) override;

				virtual void Draw(CDC& dc, CRect rcClient) {};

				virtual void MouseLClickEvent(bool bMouseDown, NP_POINT pt) {}
				virtual void MouseLDoubleClickEvent(NP_POINT pt) { MouseLClickEvent(true, pt); }
				virtual void MouseRClickEvent(bool bMouseDown, NP_POINT pt) {}
				virtual void MouseMoveEvent(NP_POINT point) {}
				virtual void ContextMenuEvent(NP_POINT point) {}

				void RectTracker(NP_POINT point, NP_2DSHAPE& rc);

				virtual void ProcessMenuCommand(studio::_menu menuID) {}
				void ShowMenu(NP_POINT point, const std::vector<studio::_menu_item>& menuList);

			protected:
				enum class _line_type{ normal, select, mouseover, hittest};
				inline Gdiplus::Color GetLineColor(_line_type type) const
				{
					switch (type)
					{
					case _line_type::mouseover:
						return m_cur_line_clr;
					case _line_type::select:
						return m_select_line_clr;
						break;
					default:
						return m_normal_line_clr;
					}
				}
				Gdiplus::REAL GetLineSize(_line_type type) const
				{
					if (type == _line_type::hittest)
						return m_hittest_line_size;
					return m_normal_line_size;
				}

				inline void DrawMappingLine(HDC hdc, _line_type line_type, const _CURVE_INTEGRATED_LINE& line, bool trans_viewport=true)
				{
					const _BEZIER_LINE* bezier;

					_BEZIER_LINE temp;
					if (trans_viewport)
					{
						line.bezier.ViewportToWnd(m_vpOrg, temp);
						bezier = &temp;
					}
					else
						bezier = &line.bezier;

					Gdiplus::Pen pen(GetLineColor(line_type), GetLineSize(line_type));
					gui::win32::GraphicUtil::CompositeLinePen(pen, line.arrow_type, line.dash_type);
					if (line.draw_type == _line_draw_type::bezier)
						GraphicUtil::DrawBezierLine(hdc, *bezier, pen);
					else
						GraphicUtil::DrawLine(hdc, bezier->start, bezier->points[bezier->points.size() - 1].pt, pen);
				}

				inline bool LineHitTest(const NP_POINT& point, const _CURVE_INTEGRATED_LINE& line) const
				{
					Gdiplus::Pen pen(GetLineColor(_line_type::hittest), GetLineSize(_line_type::hittest));
					gui::win32::GraphicUtil::CompositeLinePen(pen, line.arrow_type, line.dash_type);

					if (line.draw_type == _line_draw_type::bezier)
						return GraphicUtil::BezierLineHitTest(point, line.bezier, pen);
					else
						return GraphicUtil::LineHitTest(point, line.bezier.start, line.bezier.points[line.bezier.points.size() - 1].pt, pen);
				}

				void DrawBindingLines(HDC hdc, const _binding_link_vector& binding_link_vector, const _NEURO_BINDING_LINK* selected_link, const _NEURO_BINDING_LINK* mouseover_link);
				const _NEURO_BINDING_LINK* BindingHitTest(NP_POINT point, const _binding_link_vector& binding_link_vector) const;

				CBrush m_normal_layer_brush;
				CBrush m_select_brush;
				CBrush m_cur_layer_brush;

				Gdiplus::Color m_normal_line_clr;
				Gdiplus::Color m_select_line_clr;
				Gdiplus::Color m_cur_line_clr;

				Gdiplus::REAL m_normal_line_size;
				Gdiplus::REAL m_hittest_line_size;

				NPDropTarget m_dropTarget;
			};
		}
	}
}
