#if !defined(_JAVA_SCRIPT_RUNNER_H)
#define _JAVA_SCRIPT_RUNNER_H

#include <string>

#ifdef WIN64
#define JSRunner_DLL_Name	L"JSRunner.dll"
#else
#define JSRunner_DLL_Name	L"JSRunner32.dll"
#endif

#define JS_RUN_INSTANCE __int64

namespace np
{
	namespace js
	{
		enum class _type{ _t_int64, _t_float32, _t_float64 };

		struct _DATA
		{
			_type type;
			__int64 value;
			unsigned __int32 count;	// if array, count is size of array and value is pointer of array
		};

		struct _JS_PARAM
		{
			unsigned __int32 data_count;
			_DATA* data_vector;
		};

		struct _JS_ERROR
		{
			char err[100];
		};

		typedef JS_RUN_INSTANCE(_stdcall *_Create)(const char* script_string, _JS_ERROR* js_err);
		typedef bool(_stdcall *_Run)(JS_RUN_INSTANCE instance, const _JS_PARAM* param, _DATA* target, _JS_ERROR* js_err);
		typedef void(_stdcall *_Destroy)(JS_RUN_INSTANCE instance);

		class JavaScriptRunner
		{
		protected:
			struct _FUNCTION
			{
				_Destroy Destroy_func;
				_Run Run_func;
			};

		public:
			static JavaScriptRunner* Create(const wchar_t* strBaseDir, const char* script_string, std::string& js_err)
			{
				std::wstring path=strBaseDir;
				path+=JSRunner_DLL_Name;

				HMODULE hInstance = LoadLibraryW(path.c_str());
				if (!hInstance)
				{
					js_err = "failed to load JSRunner";
					return NULL;
				}

				_FUNCTION function;
				memset(&function, 0, sizeof(_FUNCTION));

				_Create Create_func = (_Create)GetProcAddress(hInstance, "Create");
				function.Destroy_func = (_Destroy)GetProcAddress(hInstance, "Destroy");
				function.Run_func = (_Run)GetProcAddress(hInstance, "Run");
				if (!Create_func || !function.Destroy_func || !function.Run_func)
				{
					FreeLibrary(hInstance);
					js_err = "no functions";
					return NULL;
				}

				_JS_ERROR js_error;
				JS_RUN_INSTANCE js_run_instance = Create_func(script_string, &js_error);
				if (!js_run_instance)
				{
					js_err = js_error.err;
					FreeLibrary(hInstance);
					return NULL;
				}
				
				return new JavaScriptRunner(hInstance, function, js_run_instance);
			}

			~JavaScriptRunner()
			{
				if (m_js_run_instance && m_function.Destroy_func)
					m_function.Destroy_func(m_js_run_instance);

				if (m_hInstance)
					FreeLibrary(m_hInstance);
			}

			bool Run(const _JS_PARAM& param, _DATA& target, std::string& js_err)
			{
				if (!m_function.Run_func)
				{
					js_err = "no run function";
					return false;
				}

				_JS_ERROR js_error;
				if (m_function.Run_func(m_js_run_instance, &param, &target, &js_error))
					return true;

				js_error.err[_countof(js_error.err) - 1] = 0;
				js_err = js_error.err;
				return false;
			}

		protected:
			JavaScriptRunner(HMODULE dll_instance, const _FUNCTION& function, JS_RUN_INSTANCE js_run_instance)
			{
				m_hInstance = dll_instance;
				memcpy(&m_function, &function, sizeof(_FUNCTION));
				m_js_run_instance = js_run_instance;
			}

		private:

			HMODULE m_hInstance;
			_FUNCTION m_function;

			JS_RUN_INSTANCE m_js_run_instance;
		};
	}
}

#endif
