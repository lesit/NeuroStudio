#pragma once

#include "gui/shape.h"
#include "NeuroKernel/network/HiddenLayer.h"

using namespace np::network;

class DrawLayerDesign
{
public:
	static NP_SIZE GetDrawSize();
	static NP_SIZE GetDrawShapeSize();

	static void Draw(CDC& dc, NP_RECT rc, const AbstractLayer& layer);

	class DrawShape;
	static DrawShape* CreateLayerIcon(const NP_RECT& rcDraw, const AbstractLayer& layer);

	class DrawShape
	{
	public:
		DrawShape(const NP_RECT& rcDraw, neuro_u32 shape_id, const tensor::TensorShape& ts);
		virtual ~DrawShape(){}

		void Draw(CDC& dc) const;
		virtual void ConnDraw(CDC& dc) const
		{
			Draw(dc);
		}

		virtual bool IsFilterConnect() const{ return false; }
		virtual void GetInConnPoints(_np_point_vector& points) const{};

	protected:
		virtual void DrawTensors(CDC& dc) const = 0;

		const neuro_u32 m_shape_id;
		NP_RECT m_rcDraw;
		NP_SIZE m_org_size;

		tensor::TensorShape m_ts;
	private:
		CBitmap m_bmp;
	};

	class DrawTensorShape : public DrawShape
	{
	public:
		DrawTensorShape(const NP_RECT& rcDraw, const tensor::TensorShape& ts);

		void GetOutConnPoints(bool isKenelConnect, _np_point_vector& points, NP_RECT& rcOut) const;
		void GetInConnPoints(_np_point_vector& points) const override;
	
	protected:
		void DrawTensors(CDC& dc) const override;

		neuro_u32 GetShapeID(const tensor::TensorShape& ts) const;
	};

	class DrawInputShape : public DrawTensorShape
	{
	public:
		DrawInputShape(const NP_RECT& rcDraw, const AbstractLayer& layer);
	};

	class DrawHiddenShape : public DrawShape
	{
	public:
		DrawHiddenShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id);

	protected:
		const np::nsas::_LAYER_STRUCTURE_UNION& m_entry;
	};

	class DrawFilterLayerShape : public DrawHiddenShape
	{
	public:
		DrawFilterLayerShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id);

		bool IsFilterConnect() const override{ return true; }

	};
	class DrawConvShape : public DrawFilterLayerShape
	{
	public:
		DrawConvShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

		void GetInConnPoints(_np_point_vector& points) const override;
	protected:
		void DrawTensors(CDC& dc) const override;
	};

	class DrawPoolingShape : public DrawFilterLayerShape
	{
	public:
		DrawPoolingShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

		bool IsFilterConnect() const override{ return true; }

		void GetInConnPoints(_np_point_vector& points) const override;
	protected:
		void DrawTensors(CDC& dc) const override;
	};

	class DrawNormalLayerShape : public DrawHiddenShape
	{
	public:
		DrawNormalLayerShape(const NP_RECT& rcDraw, const AbstractLayer& layer, neuro_u32 shape_id);

		void ConnDraw(CDC& dc) const override;
		void GetInConnPoints(_np_point_vector& points) const override;
	};

	class DrawFullyShape : public DrawNormalLayerShape
	{
	public:
		DrawFullyShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

	protected:
		void DrawTensors(CDC& dc) const override;
	};

	class DrawDropoutShape : public DrawNormalLayerShape
	{
	public:
		DrawDropoutShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

	protected:
		void DrawTensors(CDC& dc) const override{}
	};

	class DrawRnnShape : public DrawNormalLayerShape
	{
	public:
		DrawRnnShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

	protected:
		void DrawTensors(CDC& dc) const override{}
	};

	class DrawBatchNormShape : public DrawNormalLayerShape
	{
	public:
		DrawBatchNormShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

	protected:
		void DrawTensors(CDC& dc) const override{}
	}; 

	class DrawConcatShape : public DrawNormalLayerShape
	{
	public:
		DrawConcatShape(const NP_RECT& rcDraw, const AbstractLayer& layer);

	protected:
		void DrawTensors(CDC& dc) const override{}
	};
};
