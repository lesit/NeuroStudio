	/// \brief CPU 정보를 반환한다.
	/// \return LPCTSTR CPU 정보 문자열
	LPCTSTR GetCpuInfo()
	{
		// CPU 정보 기록
		SYSTEM_INFO	SystemInfo;
		GetSystemInfo(&SystemInfo);

		static TCHAR szCpuInfo[512] = {0,};
#ifdef UNICODE
		swprintf(szCpuInfo, _ARRAYSIZE(szCpuInfo)-1, 
			L"%d processor(s), type %d",
			SystemInfo.dwNumberOfProcessors, SystemInfo.dwProcessorType);
#else
		_snprintf_s(szCpuInfo, _ARRAYSIZE(szCpuInfo)-1, _TRUNCATE, 
			"%d processor(s), type %d",
			SystemInfo.dwNumberOfProcessors, SystemInfo.dwProcessorType);
#endif
		return szCpuInfo;
	}

	/// \brief 메모리 정보를 반환한다.
	/// \return LPCTSTR 메모리 정보 문자열
	LPCTSTR GetMemoryInfo()
	{
		static const int ONE_K = 1024;
		static const int ONE_M = ONE_K * ONE_K;
		static const int ONE_G = ONE_K * ONE_K * ONE_K;

		MEMORYSTATUS MemInfo;
		MemInfo.dwLength = sizeof(MemInfo);
		GlobalMemoryStatus(&MemInfo);

		static TCHAR szMemoryInfo[2048] = {0,};
#ifdef UNICODE
		swprintf(szMemoryInfo, _ARRAYSIZE(szMemoryInfo)-1, 
			L"%d%% of memory in use.\n"
			L"%d MB physical memory.\n"
			L"%d MB physical memory free.\n"
			L"%d MB paging file.\n"
			L"%d MB paging file free.\n"
			L"%d MB user address space.\n"
			L"%d MB user address space free.",
			MemInfo.dwMemoryLoad, 
			(MemInfo.dwTotalPhys + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailPhys + ONE_M - 1) / ONE_M, 
			(MemInfo.dwTotalPageFile + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailPageFile + ONE_M - 1) / ONE_M, 
			(MemInfo.dwTotalVirtual + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailVirtual + ONE_M - 1) / ONE_M);
#else
		_snprintf_s(szMemoryInfo, _ARRAYSIZE(szMemoryInfo)-1, _TRUNCATE,
			"%d%% of memory in use.\n"
			"%d MB physical memory.\n"
			"%d MB physical memory free.\n"
			"%d MB paging file.\n"
			"%d MB paging file free.\n"
			"%d MB user address space.\n"
			"%d MB user address space free.",
			MemInfo.dwMemoryLoad, 
			(MemInfo.dwTotalPhys + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailPhys + ONE_M - 1) / ONE_M, 
			(MemInfo.dwTotalPageFile + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailPageFile + ONE_M - 1) / ONE_M, 
			(MemInfo.dwTotalVirtual + ONE_M - 1) / ONE_M, 
			(MemInfo.dwAvailVirtual + ONE_M - 1) / ONE_M);
#endif

		return szMemoryInfo;
	}