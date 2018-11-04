#if !defined(_FILE_ADAPTOR_H)
#define _FILE_ADAPTOR_H

#include "platform_env.h"

#include "DeviceAdaptor.h"
/*
#include "CacheReadBuffer.h"
#include "CacheWriteBuffer.h"
*/
namespace np
{
	namespace device
	{
		class FileDeviceAdaptor : public DeviceAdaptor
		{
		public:
			FileDeviceAdaptor();
			virtual ~FileDeviceAdaptor();

			virtual _device_type GetType() const {return _device_type::file;}

			virtual bool Create(const char* strFilePath, bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size = 0);
			virtual void Close();

			FILE_HANDLE GetFileHandle(){return m_hFile;}

			std::string GetDeviceName() const override;
			std::string GetDevicePath() const override{
				return m_file_path;
			}

			const char* GetFilePath() const{ return m_file_path.c_str(); }

			virtual neuro_u64 GetMaxExtensibleSize() const;

			virtual bool SetPosition(neuro_u64 pos) override;
			virtual neuro_u64 GetPosition() const override;
			virtual neuro_u32 Read(void* buffer, neuro_u32 nSize) const override;
			virtual neuro_u32 Write(const void* buffer, neuro_u32 nSize) override;

			bool Flush() override;

			neuro_u64 GetUsageSize() const override;
		protected:
			FILE_HANDLE m_hFile;

			std::string m_file_path;

//			device::CCacheReadBuffer m_read;
//			device::CCacheWriteBuffer m_write;
		};

		class FileDeviceFactory : public IODeviceFactory
		{
		public:
			FileDeviceFactory(const char* strFilePath);
			virtual ~FileDeviceFactory();

			virtual _device_type GetType() const {return _device_type::file;}

			virtual void Reset();

			bool operator == (const IODeviceFactory& right) const;

			const char* GetFilePath() const { return m_file_path.c_str(); }

		protected:
			virtual DeviceAdaptor* Create(bool bReadOnly, bool bCreateAlways, bool bShareWrite, neuro_u64 init_size);

			std::string m_file_path;
		};
	}
}

#endif
