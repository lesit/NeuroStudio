#pragma once

#include "AbstractProducerModel.h"
#include "gui/shape.h"

namespace np
{
	namespace dp
	{
		namespace model
		{
			class MnistImageProducerModel : public ImageProcessingProducerModel
			{
			public:
				MnistImageProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: ImageProcessingProducerModel(provider, uid)
				{}
				virtual ~MnistImageProducerModel() {}

				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::no; }
				_input_source_type GetInputSourceType() const override {
					return _input_source_type::mnist_img_file;
				}

				_producer_type GetProducerType() const override {
					return _producer_type::mnist_img;
				}
				bool AvailableToOutputLayer() const override { return false; }

				bool AvailableUsingPredict() const { return false; }

				tensor::DataShape GetDataShape() const override {
					return tensor::DataShape({ 1, 28, 28 });
				}
			};

			class MnistLabelProducerModel : public ImageProcessingProducerModel
			{
			public:
				MnistLabelProducerModel(DataProviderModel& provider, neuro_u32 uid)
					: ImageProcessingProducerModel(provider, uid)
				{}
				virtual ~MnistLabelProducerModel() {}

				_has_input_reader_status HasInputReaderStatus() const override { return _has_input_reader_status::no; }
				_input_source_type GetInputSourceType() const override {
					return _input_source_type::mnist_label_file;
				}

				_producer_type GetProducerType() const override {
					return _producer_type::mnist_label;
				}
				bool AvailableToInputLayer() const override { return false; }

				bool AvailableUsingPredict() const { return false; }

				tensor::DataShape GetDataShape() const override {
					return tensor::DataShape({ 1, 1, 10 });
				}

				_label_out_type GetLabelOutType() const override { return _label_out_type::direct_def; }
				neuro_u32 GetLabelOutCount() const override {
					return 10;
				}
				void SetLabelOutCount(neuro_u32 scope) override { }

			};
		}
	}
}
