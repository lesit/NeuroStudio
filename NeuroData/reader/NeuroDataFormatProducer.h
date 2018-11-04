#pragma once

#include "AbstractProducer.h"
#include "storage/DeviceAdaptor.h"
#include "tensor/tensor_shape.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{

#pragma pack(push, 1)
			const unsigned char mark_ndf[] = { 0x13, 0x05, 0x07 };
			const neuro_u32 ndf_version_0_0_0_1 = 0x00000001L;
			const neuro_u32 ndf_version_0_0_0_1_header_size = 336;

			const neuro_u32 ndf_version_0_0_0_2 = 0x00000002L;

			union _NDF_HEADER
			{
				struct
				{
					unsigned char mark[_countof(mark_ndf)];	// mark must be mark_neuroformatdata
					neuro_u32 version;

					neuro_u32 header_size;				// header_size = sizeof(_NDF_HEADER)

					char source_device_name[256];
					neuro_u32 origin_producer_type;
					neuro_u32 origin_producer_info;

					model::_ndf_dim_type dim_type;
					_TYPED_VECTOR_N_DEF<neuro_u32, 10> data_shape;

					neuro_size_t data_count;				// count of data

					neuro_size_t total_value_count;
				};
				char reserved[1024];
			};

			struct _NDF_ROW_DATA
			{
				neuro_u32 sub_row_count;	// sub_row_count <= data_shape[1 ~ ]
			};

			/* neuro format data type
				_NP_FORMAT_HEADER

				column : row1 count, row1
				row1 : row2 count, row2
				row2 : row3 count, row3
			*/
#pragma pack(pop)

			class NeuroDataFormatProducer : public AbstractProducer
			{
			public:
				static NeuroDataFormatProducer* GetExistNdfCloneProducer(DataReaderSet& reader_set, const model::AbstractProducerModel& original_model);
				static NeuroDataFormatProducer* CreateNewNdfCloneProducer(DataReaderSet& reader_set, const model::AbstractProducerModel& original_model);

				static bool ReadHeader(device::DeviceAdaptor& device, _NDF_HEADER& header);

				virtual ~NeuroDataFormatProducer();

				bool Create(DataReaderSet& reader_set) override { return false; }

				virtual const wchar_t* GetTypeString() const { return L"NeuroDataFormatProducer"; }

				neuro_size_t GetRawDataCount() const override {
					return m_header.data_count;
				}

			protected:
				NeuroDataFormatProducer(const _NDF_HEADER& header, const tensor::DataShape& data_shape);
				bool AttachNdfDevice(device::DeviceAdaptor* input_device);

				neuro_u32 ReadRawData(neuro_size_t pos, neuron_value* value, bool is_ndf_mode) override;
				void DataNoising(neuron_value* value) override;

				void ReadRow(neuron_value*& buffer, int depth, neuron_value*& value);

#if 0//!defined(_NDF_CHECK_ORG_PRODUCER)
#define _NDF_CHECK_ORG_PRODUCER
#endif

#ifdef _NDF_CHECK_ORG_PRODUCER
				AbstractProducer* m_check_org_producer;	// 검증을 위한 소스
#endif

				device::DeviceAdaptor* m_input_device;

				_NDF_HEADER m_header;

				tensor::DataShape m_read_data_shape;

				neuro_8 m_last_shape_index;

				neuro_size_t* m_index_table;

				neuro_size_t m_device_size;

				neuro_u8* m_read_buffer;
				neuro_size_t m_buffer_size;

				neuro_size_t m_device_start_position;
				neuro_size_t m_device_cur_position;

				neuro_size_t m_max_rows;
			};
		}
	}
}
