#if !defined(_MEMORY_MAPPED_FILE_ADAPTOR)
#define _MEMORY_MAPPED_FILE_ADAPTOR

#include "FileDeviceAdaptor.h"

namespace np
{
	namespace device
	{
		class MMFDeviceAdaptor : public FileDeviceAdaptor
		{
		public:
			MMFDeviceAdaptor();
			virtual ~MMFDeviceAdaptor();

			virtual bool Create(const char* strFilePath, bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size = 0) override;
			virtual bool SetUsageSize(neuro_u64 new_size) override;

			virtual bool SetPosition(neuro_u64 pos) override;
			virtual neuro_u64 GetPosition() const override{
				return m_position;
			}
			virtual neuro_u32 Read(void* buffer, neuro_u32 nSize) const override;
			virtual neuro_u32 Write(const void* buffer, neuro_u32 nSize) override;

			bool Flush() override;

		protected:
			neuro_u8* GetMemoryBuffer(neuro_u32 nSize, neuro_u32& nUsable);

			neuro_u64 m_nFileSize;
			FILE_HANDLE m_hFileMapping;
			bool m_bReadOnly;

			unsigned long m_dwAllocationGranularity;

			neuro_u8* m_pbFile;

			const unsigned long m_nMaxBlockSize;

			neuro_u64 m_nMappedPointer;
			unsigned long m_nMappedSize;

			neuro_u64 m_position;
		};

		class MMFDeviceFactory : public FileDeviceFactory
		{
		public:
			MMFDeviceFactory(const char* strFilePath)
				: FileDeviceFactory(strFilePath)	{}
			virtual ~MMFDeviceFactory(){};

		protected:
			virtual DeviceAdaptor* Create(bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size)
			{
				MMFDeviceAdaptor* device=new MMFDeviceAdaptor;
				if(!device)
					return NULL;
				if (device->Create(m_file_path.c_str(), bReadOnly, bCreateAlways, bShareWrite, init_size))
					return device;

				delete device;
				return NULL;
			}
		};
	}
}

#endif
