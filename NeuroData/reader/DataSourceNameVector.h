#pragma once

#include "util/UniqueIdFactory.h"
#include "util/FileUtil.h"

namespace np
{
	namespace dp
	{
		namespace preprocessor
		{
			template<typename T = char>
			class DataSourceBasePath
			{
			public:
				virtual ~DataSourceBasePath() {}

				DataSourceBasePath& operator=(const DataSourceBasePath<T>& src)
				{
					m_id_factory = src.m_id_factory;

					m_basedir_id_map = src.m_basedir_id_map;
					m_id_basedir_map = src.m_id_basedir_map;
					return *this;
				}

				typedef std::basic_string<T> _string;
				neuro_u32 GetId(_string base_path)
				{
					if (base_path.empty())
						return neuro_last32;

					if (base_path.back() == '\\' || base_path.back() == '/')
						base_path.erase(base_path.end() - 1);

					_string trans_dir;
#if defined(_WINDOWS)
					trans_dir = util::StringUtil::ToLower(base_path);
#else
					trans_dir = base_path;
#endif
					neuro_u32 base_path_id = neuro_last32;

					std::unordered_map<_string, neuro_u32>::const_iterator it = m_basedir_id_map.find(trans_dir);
					if (it == m_basedir_id_map.end())
					{
						base_path_id = m_id_factory.CreateId();
						m_basedir_id_map.insert({ trans_dir, base_path_id });
						m_id_basedir_map[base_path_id] = base_path;
					}
					else
						base_path_id = it->second;
					return base_path_id;
				}

				_string GetPath(neuro_u32 id) const
				{
					std::unordered_map<neuro_u32, _string>::const_iterator it = m_id_basedir_map.find(id);
					if (it == m_id_basedir_map.end())
						return _string();

					return it->second;
				}
			private:
				util::UniqueIdFactory m_id_factory;

				std::unordered_map<_string, neuro_u32> m_basedir_id_map;
				std::unordered_map<neuro_u32, _string> m_id_basedir_map;
			};

			template<typename T = char>
			struct _DATA_SOURCE_NAME
			{
				_DATA_SOURCE_NAME()
				{
					base_path_id = neuro_last32;
					data = 0;
				}

				_DATA_SOURCE_NAME& operator=(const _DATA_SOURCE_NAME& src)
				{
					base_path_id = src.base_path_id;
					name = src.name;
				}
				neuro_u32 base_path_id;
				std::basic_string<T> name;

				neuro_u64 data;
			};

			template<typename T = char>
			class DataSourceNameVector
			{
			public:
				virtual ~DataSourceNameVector() {}

				DataSourceNameVector& operator=(const DataSourceNameVector<T>& src)
				{
					m_basedir = src.m_basedir;
					m_name_vector = src.m_name_vector;
					return *this;
				}

				typedef std::basic_string<T> _string;

				void AddPath(const T* path, neuro_u64 data = 0)
				{
					AddPath(util::FileUtil::GetDirFromPath<T>(path).c_str(), util::FileUtil::GetFileName<T>(path), data);
				}

				void AddPath(const T* base_path, const T* name, neuro_u64 data = 0)
				{
					_DATA_SOURCE_NAME<T> data_source_name;
					data_source_name.base_path_id = m_basedir.GetId(base_path);
					if (data_source_name.base_path_id == neuro_last32)
						return;

					data_source_name.name = name;
					data_source_name.data = data;
					m_name_vector.push_back(data_source_name);
				}

				neuro_size_t GetCount() const { return m_name_vector.size(); }

				_string GetName(neuro_size_t i) const
				{
					if (i >= m_name_vector.size())
						return "";

					return m_name_vector[i].name;
				}

				_string GetPath(neuro_size_t i) const
				{
					if (i >= m_name_vector.size())
						return _string();

					const _DATA_SOURCE_NAME<T>& data_source_name = m_name_vector[i];

					_string ret = m_basedir.GetPath(data_source_name.base_path_id);
					if (ret.empty())
						return ret;
#if defined(_WINDOWS)
					ret.append("\\");
#else
					ret+='/';
#endif
					return ret.append(data_source_name.name);
				}

				neuro_u64 GetData(neuro_size_t i) const
				{
					if (i >= m_name_vector.size())
						return 0;
					return m_name_vector[i].data;
				}

			private:
				DataSourceBasePath<T> m_basedir;

				std::vector<_DATA_SOURCE_NAME<T>> m_name_vector;
			};
			typedef std::unordered_map<neuro_u32, DataSourceNameVector<char>> _uid_datanames_map;
			typedef std::unordered_map<neuro_u32, DataSourceNameVector<wchar_t>> _uid_wchar_datanames_map;
		}

	}
}
