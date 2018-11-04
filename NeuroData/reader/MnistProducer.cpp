#include "stdafx.h"

#include "MnistProducer.h"

using namespace np::dp;
using namespace np::dp::preprocessor;

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			namespace mnist
			{
				struct mnist_header {
					neuro_u32 magic_number;
					neuro_u32 num_items;
					neuro_u32 num_rows;
					neuro_u32 num_cols;
				};

				struct mnist_label_header {
					neuro_u32 magic_number;
					neuro_u32 num_items;
				};

				inline bool read_mnist_header(const device::DeviceAdaptor& device, mnist_header& header)
				{
					if (device.Read(&header, sizeof(mnist_header)) != sizeof(mnist_header))
						return false;

					if (is_little_endian())
					{
						reverse_endian(&header.magic_number);
						reverse_endian(&header.num_items);
						reverse_endian(&header.num_rows);
						reverse_endian(&header.num_cols);
					}

					if (header.magic_number != 0x00000803 || header.num_items <= 0)
						return false;

					return true;
				}

				inline bool read_mnist_image(const device::DeviceAdaptor& device,
					const neuro_u32 mnist_width,
					const neuro_u32 mnist_height,
					neuro_u8* temp_read_buffer,
					const neuro_u32 read_width,
					const neuro_u32 read_height,
					const neuron_value scale_min,
					const neuron_value scale_max,
					neuron_value* buffer)
				{
					if (read_width < mnist_width || read_height < mnist_height)
						return false;
					neuro_u32 x_padding = read_width - mnist_width;
					neuro_u32 y_padding = read_height - mnist_height;

					if (device.Read(temp_read_buffer, mnist_height * mnist_width) != mnist_height * mnist_width)
						return false;

					const neuron_value* last_buffer = buffer + read_width*read_height;

					const neuron_value scale = (scale_max - scale_min) / neuron_value(255);

					if (x_padding == 0 && y_padding == 0)
					{
						for (neuron_value* bits = buffer; bits < last_buffer; bits++, temp_read_buffer++)	// padding
							*bits = *temp_read_buffer * scale + scale_min;

						/*#ifdef _DEBUG
											DEBUG_OUTPUT(L"mnst image--->");
											NP_Util::DebugOutputValues(buffer, read_width*read_height, 10);
											DEBUG_OUTPUT(L"<--- mnst image\r\n");
						#endif*/
					}
					else
					{
						neuro_u32 left_padding = x_padding / 2;
						neuro_u32 right_padding = x_padding - left_padding;
						neuro_u32 top_padding = y_padding / 2;
						neuro_u32 bottom_padding = y_padding - top_padding;

						// upper padding
						neuron_value* bits = buffer;
						neuron_value* end_bits = bits + top_padding * read_width;
						for (; bits < end_bits; bits++)
							*bits = scale_min;

						for (neuro_u32 y = 0; y < mnist_height; y++)
						{
							// left padding
							for (neuro_u32 i = 0; i < left_padding; i++, bits++)
								*bits = scale_min;

							for (neuro_u32 i = 0; i < mnist_width; i++, bits++, temp_read_buffer++)
							{
								*bits = *temp_read_buffer * scale + scale_min;
							}

							// right padding
							for (neuro_u32 i = 0; i < right_padding; i++, bits++)
								*bits = scale_min;
						}

						// lower padding
						bits = buffer + (read_height - top_padding)*read_width;
						end_bits = bits + bottom_padding * read_width;
						for (; bits < end_bits; bits++)
							*bits = scale_min;

						bool b = bits == last_buffer;
					}

					return true;
				}

				inline bool read_mnist_label_header(const device::DeviceAdaptor& device, neuro_u32& count)
				{
					mnist_label_header header;
					if (device.Read(&header, sizeof(mnist_label_header)) != sizeof(mnist_label_header))
						return false;

					if (is_little_endian())
					{
						reverse_endian(&header.magic_number);
						reverse_endian(&header.num_items);
					}

					if (header.magic_number != 0x00000801 || header.num_items <= 0)
						return false;

					count = header.num_items;
					return true;
				}

				inline neuro_size_t parse_mnist_labels(const device::DeviceAdaptor& device, neuro_u8* labels, neuro_size_t count)
				{
					device.Read(labels, (neuro_u32)count);
					return count;
				}
			}
		}
	}
}

MnistProducer::MnistProducer(const model::AbstractProducerModel& model)
	: AbstractProducer(model)
{
	m_temp_read_buffer = NULL;
	m_data_count = 0;
}

MnistProducer::~MnistProducer()
{
	if (m_temp_read_buffer)
		free(m_temp_read_buffer);
}

void MnistProducer::OnPreloadCompleted()
{
	if (m_temp_read_buffer)
		free(m_temp_read_buffer);
	m_temp_read_buffer = NULL;
}

MnistImageProducer::MnistImageProducer(const model::AbstractProducerModel& model)
: MnistProducer(model), m_model((const model::MnistImageProducerModel&)model)
{
	m_data_count = 0;
}

MnistImageProducer::~MnistImageProducer()
{
}

bool MnistImageProducer::Create(DataReaderSet& reader_set)
{
	if (!AttachInputDevices(m_model, reader_set))
		return false;

	mnist::mnist_header header;
	if(!mnist::read_mnist_header(*m_device_vector[0], header))
		return false;

	if (m_data_shape.GetWidth() < header.num_cols || m_data_shape.GetHeight() < header.num_rows)
		return false;

	m_img_sz.width = header.num_cols;
	m_img_sz.height = header.num_rows;
	m_temp_read_buffer = (neuro_u8*)malloc(m_img_sz.width * m_img_sz.height);

	m_data_count = header.num_items;;

	m_device_vector[0]->SetPosition(sizeof(mnist::mnist_header));
	return true;
}

neuro_u32 MnistImageProducer::ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode)
{
	if (pos >= m_data_count)
		return 0;

	device::DeviceAdaptor* device = m_device_vector[0];
	if(!device->SetPosition(sizeof(mnist::mnist_header) + pos * m_data_dim_size))
		return 0;

	if (!mnist::read_mnist_image(*device
		, m_img_sz.width, m_img_sz.height
		, m_temp_read_buffer
		, m_data_shape.GetWidth(), m_data_shape.GetHeight()
		, m_scale_min, m_scale_max
		, value))
		return 0;

	return m_data_dim_size;
}


MnistLabelProducer::MnistLabelProducer(const model::AbstractProducerModel& model)
: MnistProducer( model), m_model((const model::MnistLabelProducerModel&)model)
{
}

MnistLabelProducer::~MnistLabelProducer()
{
}

bool MnistLabelProducer::Create(DataReaderSet& reader_set)
{
	if (!AttachInputDevices(m_model, reader_set))
		return false;

	if (!mnist::read_mnist_label_header(*m_device_vector[0], m_data_count))
		return false;

	if (!m_device_vector[0]->SetPosition(sizeof(mnist::mnist_label_header)))
		return false;
	
	m_temp_read_buffer = (neuro_u8*)malloc(m_data_count);
	if(m_device_vector[0]->Read(m_temp_read_buffer, m_data_count) != m_data_count)
	{
		free(m_temp_read_buffer);
		m_temp_read_buffer = NULL;
		return false;
	}
	return true;
}

bool MnistLabelProducer::ReadRawLabel(neuro_size_t pos, neuro_u32& label)
{
	if (pos >= m_data_count)
		return false;

	label = m_temp_read_buffer[pos];
	return true;
}
