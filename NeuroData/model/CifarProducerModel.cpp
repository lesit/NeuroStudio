#include "CifarProducerModel.h"

using namespace np::dp;
using namespace np::dp::model;

#define CIFAR10_IMAGE_DEPTH (3)
#define CIFAR10_IMAGE_WIDTH (32)
#define CIFAR10_IMAGE_HEIGHT (32)
#define CIFAR10_IMAGE_AREA (CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT)
#define CIFAR10_IMAGE_SIZE (CIFAR10_IMAGE_AREA*CIFAR10_IMAGE_DEPTH)

namespace np
{
	namespace dp
	{
		namespace cfar
		{
			/*
			inline void parse_cifar10(const device::DeviceAdaptor& device,
				std::vector<vec_t> *train_images,
				std::vector<label_t> *train_labels,
				neuron_value scale_min,
				neuron_value scale_max,
				int x_padding,
				int y_padding)
			{
				if (x_padding < 0 || y_padding < 0)
					throw nn_error("padding size must not be negative");
				if (scale_min >= scale_max)
					throw nn_error("scale_max must be greater than scale_min");

				std::ifstream ifs(filename, std::ios::in | std::ios::binary);
				if (ifs.fail() || ifs.bad())
					throw nn_error("failed to open file:" + filename);

				uint8_t label;
				std::vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);

				while (ifs.read((char*)&label, 1)) {
					vec_t img;

					if (!ifs.read((char*)&buf[0], CIFAR10_IMAGE_SIZE)) break;

					if (x_padding || y_padding)
					{
						int w = CIFAR10_IMAGE_WIDTH + 2 * x_padding;
						int h = CIFAR10_IMAGE_HEIGHT + 2 * y_padding;

						img.resize(w * h * CIFAR10_IMAGE_DEPTH, scale_min);

						for (int c = 0; c < CIFAR10_IMAGE_DEPTH; c++) {
							for (int y = 0; y < CIFAR10_IMAGE_HEIGHT; y++) {
								for (int x = 0; x < CIFAR10_IMAGE_WIDTH; x++) {
									img[c * w * h + (y + y_padding) * w + x + x_padding]
										= scale_min + (scale_max - scale_min) * buf[c * CIFAR10_IMAGE_AREA + y * CIFAR10_IMAGE_WIDTH + x] / 255;
								}
							}
						}
					}
					else
					{
						std::transform(buf.begin(), buf.end(), std::back_inserter(img),
							[=](unsigned char c) { return scale_min + (scale_max - scale_min) * c / 255; });
					}

					train_images->push_back(img);
					train_labels->push_back(label);
				}
			}
			*/
		}
	}
}

CifarProducerModel::CifarProducerModel(DataProviderModel& provider, neuro_u32 uid)
	: ImageProcessingProducerModel(provider, uid)
{
}


CifarProducerModel::~CifarProducerModel()
{
}
