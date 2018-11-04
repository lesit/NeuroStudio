#pragma once

#include "np_types.h"
#include "util/np_util.h"

namespace np
{
	namespace dp
	{
		class JobSignalReciever
		{
		public:
			virtual void dataio_long_time_job_start(neuro_u32 job_id, const char* status){ dataio_job_status(job_id, status); }
			virtual void dataio_long_time_job_end(neuro_u32 job_id, const char* status, double elapse){ dataio_job_status(job_id, status); }
			virtual void dataio_failure_job(neuro_u32 job_id, const char* status){ dataio_job_status(job_id, status); }

			virtual void dataio_job_status(neuro_u32 job_id, const char* status) = 0;
		};

		class JobSignalSender
		{
		public:
			JobSignalSender(JobSignalReciever* signal, neuro_u32 job_id, const std::string& job_name)
				: m_job_id(job_id), m_job_name(job_name)
			{
				std::string status = std::string("start ").append(m_job_name);
				DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(status).c_str());

				m_signal = signal;
				if (m_signal)
					m_signal->dataio_long_time_job_start(m_job_id, status.c_str());
			}
			virtual ~JobSignalSender()
			{
				std::string status = std::string("completed ").append(m_job_name);
				DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(status).c_str());

				if (m_signal)
					m_signal->dataio_long_time_job_end(m_job_id, status.c_str(), m_timer.elapsed());
			}

			void current_status(std::string status)
			{
				status = std::string(m_job_name).append(" : ").append(status);
				DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(status).c_str());

				if (m_signal)
					m_signal->dataio_job_status(m_job_id, status.c_str());
			}

			void failure(const std::string& reason="")
			{
				std::string status = "failed ";
				status.append(m_job_name);
				if (!reason.empty())
					status.append(" : ").append(reason);
				DEBUG_OUTPUT(util::StringUtil::MultiByteToWide(status).c_str());
				if (m_signal)
				{
					m_signal->dataio_failure_job(m_job_id, status.c_str());
				}
				m_signal = NULL;
			}
			JobSignalReciever* m_signal;
			Timer m_timer;
			const neuro_u32 m_job_id;
			std::string m_job_name;
		};
	}
}
