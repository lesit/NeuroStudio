#include "stdafx.h"

#include "NeuroEntryAccess.h"
#include "NeuroDataAccessManager.h"

using namespace np;
using namespace np::nsas;

NetworkEntriesAccessor::NetworkEntriesAccessor(NeuroStorageAllocationSystem& nsas)
	: m_nsas(nsas)
	, m_input_layer_nda(nsas, nsas.GetNNDef().input_layer_nda)
	, m_hidden_layer_nda(nsas, nsas.GetNNDef().hidden_layer_nda)
{

}

NetworkEntriesReader::NetworkEntriesReader(NeuroStorageAllocationSystem& nsas)
	: NetworkEntriesAccessor(nsas)
{

}

neuro_u32 NetworkEntriesReader::GetInputLayerCount() const
{
	return m_input_layer_nda.GetSize() / sizeof(_INPUT_LAYER_ENTRY);
}

bool NetworkEntriesReader::ReadInputLayer(_INPUT_LAYER_ENTRY& entry)
{
	if (!m_input_layer_nda.ReadData(&entry, sizeof(_INPUT_LAYER_ENTRY)))
	{
		DEBUG_OUTPUT(L"failed read input layer\r\n");
		return false;
	}

	return true;
}

neuro_u32 NetworkEntriesReader::GetHiddenLayerCount() const
{
	return m_hidden_layer_nda.GetSize() / sizeof(_HIDDEN_LAYER_ENTRY);
}

bool NetworkEntriesReader::ReadHiddenLayer(_HIDDEN_LAYER_ENTRY& entry)
{
	if (!m_hidden_layer_nda.ReadData(&entry, sizeof(_HIDDEN_LAYER_ENTRY)))
	{
		DEBUG_OUTPUT(L"failed read layer entry");
		return false;
	}

	return true;
}

NetworkEntriesWriter::NetworkEntriesWriter(NeuroStorageAllocationSystem& nsas)
	: NetworkEntriesAccessor(nsas)
	, m_nn_def(nsas.GetNNDef())
	, m_data_nda_bitmap(nsas, nsas.GetNNDef().nda_table.bitmap_spec)
	, m_data_nda_table(nsas, nsas.GetNNDef().nda_table.nda_spec)
	, m_block_size(nsas.GetBlockSize())
	, m_nU32PerBlock(nsas.GetBlockSize() / sizeof(neuro_u32))
{
	m_block_data = (neuro_u8*)malloc(m_block_size);
}

NetworkEntriesWriter::~NetworkEntriesWriter()
{
	if(m_block_data)
		free(m_block_data);

	m_input_layer_nda.DeleteFromCurrent();
	m_hidden_layer_nda.DeleteFromCurrent();
}

bool NetworkEntriesWriter::LayersDeleted(const _layer_data_nid_vector& data_vector)
{
	for (size_t i = 0; i < data_vector.size(); i++)
	{
		const _LAYER_DATA_NID_SET& nid_set = data_vector[i];

		_std_u32_vector nid_vector;
		if(nid_set.add_input_nid!=neuro_last32)
			nid_vector.push_back(nid_set.add_input_nid);

		for (int i = 0; i < nid_set.data_nids.nid_count; i++)
			nid_vector.push_back(nid_set.data_nids.nid_vector[i]);

		if(nid_vector.size()>0)
			DeallocDataNda(nid_vector.size(), nid_vector.data());
	}
	return true;
}

bool NetworkEntriesWriter::WriteInputLayer(const _INPUT_LAYER_ENTRY& entry)
{
	if (!m_input_layer_nda.WriteData(&entry, sizeof(_INPUT_LAYER_ENTRY)))
	{
		DEBUG_OUTPUT(L"failed read input layer[%u]\r\n", entry.uid);
		return false;
	}

	return true;
}

bool NetworkEntriesWriter::WriteHiddenLayer(const _HIDDEN_LAYER_ENTRY& entry)
{
	if (!m_hidden_layer_nda.WriteData(&entry, sizeof(_HIDDEN_LAYER_ENTRY)))
	{
		DEBUG_OUTPUT(L"failed read layer entry[%u]", entry.uid);
		return false;
	}

	return true;
}

bool NetworkEntriesWriter::AllocDataNda(neuro_u32 n, neuro_u32* nid_list)
{
	if (n == 0)
		return true;

	const neuro_u64 old_nda_bitmap_size = m_data_nda_bitmap.GetSize();
	const neuro_u32 old_block = old_nda_bitmap_size / m_block_size;

	const neuro_u64 old_nda_table_size = m_data_nda_table.GetSize();

	neuro_u32* p32BitmapBlock = (neuro_u32*)m_block_data;

	// 만약 현재 남아 있는게 없으면 일단 필요한 만큼 bitmap을 확장한 후에 할당하니까, 할당할땐 확장전 마지막 포인터부터 찾으면 더 빠르다.
	neuro_u32 start_bitmap_block = 0;
	if (m_nn_def.nda_table.free_nda_count == 0)
		start_bitmap_block = m_data_nda_bitmap.GetSize() / m_block_size;

	if (n > m_nn_def.nda_table.free_nda_count)
	{
		neuro_u32 add_block = NP_Util::CalculateCountPer(n - m_nn_def.nda_table.free_nda_count, m_block_size * 8); // 1byte당 8개를 표현할 수 있으므로..
		const neuro_u32 add_nda = add_block * m_block_size * 8;

		neuro_u64 prev_pointer = m_data_nda_bitmap.GetSize();
		// 어짜피 블록개수 만큼 I/O를 하니까
		if (!m_data_nda_bitmap.SetSize((old_block + add_block) * m_block_size))
		{
			DEBUG_OUTPUT(L"failed increse data nda bitmap size %llu -> %llu", old_nda_bitmap_size, (old_block + add_block) * m_block_size);
			return false;
		}

		m_data_nda_bitmap.SetDataPointer(prev_pointer);

		memset(m_block_data, 0, m_block_size);
		for (neuro_u32 i = 0; i < add_block; i++)
		{
			if (!m_data_nda_bitmap.WriteData(m_block_data, m_block_size))
			{
				DEBUG_OUTPUT(L"failed write data nda table bitmap");
				m_data_nda_bitmap.SetSize(old_nda_bitmap_size);
				return false;
			}
		}

		m_nn_def.nda_table.free_nda_count += add_nda;
	}

	_NEURO_DATA_ALLOC_SPEC zero_spec;
	memset(&zero_spec, 0, sizeof(_NEURO_DATA_ALLOC_SPEC));

	neuro_u32 nid_index = 0;

	m_data_nda_bitmap.SetDataPointer(start_bitmap_block * m_block_size);
	for (int block = start_bitmap_block, block_count = m_data_nda_bitmap.GetSize() / m_block_size; block < block_count && nid_index < n; block++)
	{
		const neuro_u64 bitmap_pointer = m_data_nda_bitmap.GetDataPointer();
		if (!m_data_nda_bitmap.ReadData(p32BitmapBlock, m_block_size))
		{
			DEBUG_OUTPUT(L"failed read data nda table bitmap");
			m_data_nda_bitmap.SetSize(old_nda_bitmap_size);
			m_data_nda_table.SetSize(old_nda_table_size);
			return false;
		}

		neuro_u32 start_nid = block * m_block_size * 8;

		bool isUpdate = false;
		for (int i = 0; i < m_nU32PerBlock && nid_index < n; i++, start_nid+=32)
		{
			if (p32BitmapBlock[i] == 0xFFFFFFFF)
				continue;

			neuro_u32 value = p32BitmapBlock[i];

			for (int bit = 0; bit < 32 && nid_index < n; bit++)
			{
				neuro_u32 shift = 1 << bit;
				if ((shift & value) != 0)
					continue;

				value |= shift;// block 할당

				--m_nn_def.nda_table.free_nda_count;

				const neuro_u32 nid = start_nid + bit;
				nid_list[nid_index++] = nid;

				if (!m_data_nda_table.SetDataPointer(nid * sizeof(_NEURO_DATA_ALLOC_SPEC)))
				{
					DEBUG_OUTPUT(L"the nid[%u] position is not allocated. nda table size is %llu", nid, m_data_nda_table.GetSize()/ sizeof(_NEURO_DATA_ALLOC_SPEC));
					m_data_nda_bitmap.SetSize(old_nda_bitmap_size);
					m_data_nda_table.SetSize(old_nda_table_size);
					return false;
				}
				if (!m_data_nda_table.WriteData(&zero_spec, sizeof(_NEURO_DATA_ALLOC_SPEC)))
				{
					DEBUG_OUTPUT(L"failed nid[%u]", nid);
					m_data_nda_bitmap.SetSize(old_nda_bitmap_size);
					m_data_nda_table.SetSize(old_nda_table_size);
					return false;
				}
			}
			if (p32BitmapBlock[i] != value)
			{
				p32BitmapBlock[i] = value;
				isUpdate = true;
			}
		}
		if (isUpdate)
		{
			m_data_nda_bitmap.SetDataPointer(bitmap_pointer);
			m_data_nda_bitmap.WriteData(p32BitmapBlock, m_block_size);
		}
	}
	if (nid_index < n)
	{
		DEBUG_OUTPUT(L"not allocated[%u] all nids[%u]", nid_index, n);
		m_data_nda_bitmap.SetSize(old_nda_bitmap_size);
		m_data_nda_table.SetSize(old_nda_table_size);
		return false;
	}
	return true;
}

void NetworkEntriesWriter::DeallocDataNda(neuro_u32 n, neuro_u32* nid_list)
{
	const neuro_u32 nid_table_nid_count = m_data_nda_table.GetSize() / sizeof(_NEURO_DATA_ALLOC_SPEC);
	for (int i = 0; i < n; i++)
	{
		neuro_u32 nid = nid_list[i];
		if (nid >= nid_table_nid_count)
		{
			DEBUG_OUTPUT(L"nid[%u] is invalid. over nid count[%u]", nid, nid_table_nid_count);
			continue;
		}
		const neuro_u32 bit_index = nid / 8;
		m_data_nda_bitmap.SetDataPointer(bit_index);

		neuro_u8 bitmap;
		m_data_nda_bitmap.ReadData(&bitmap, sizeof(neuro_u8));

		neuro_u8 bits = 1 << (nid % 8);
		if ((bitmap & bits) == 0)
			continue;

		bitmap &= ~bits;
		m_data_nda_bitmap.SetDataPointer(bit_index);
		m_data_nda_bitmap.WriteData(&bitmap, sizeof(neuro_u8));

		m_data_nda_table.SetDataPointer(nid * sizeof(_NEURO_DATA_ALLOC_SPEC));

		_NEURO_DATA_ALLOC_SPEC spec;
		if (m_data_nda_table.ReadData(&spec, sizeof(_NEURO_DATA_ALLOC_SPEC)))
		{
			NeuroDataAccessManager nda(m_nsas, spec);
			nda.SetSize(0);
		}
		if (nid == (nid_table_nid_count - 1))	// 필요 없는건 줄이도록 하자
			m_data_nda_table.SetSize(nid * sizeof(_NEURO_DATA_ALLOC_SPEC));

		++m_nn_def.nda_table.free_nda_count;
	}
}

NeuroDataNodeAccessor::NeuroDataNodeAccessor(NeuroStorageAllocationSystem& nsas, neuro_u32 nid, bool is_readonly)
	: m_data_nda_table(nsas, nsas.GetNNDef().nda_table.nda_spec), m_writing_nid(is_readonly ? neuro_last32 : nid)
{
	m_data_nda = Open(nsas, nid);
}

NeuroDataNodeAccessor::~NeuroDataNodeAccessor()
{
	delete m_data_nda;

	if (m_writing_nid!= neuro_last32)
	{
		if(!m_data_nda_table.SetDataPointer(m_writing_nid * sizeof(_NEURO_DATA_ALLOC_SPEC)))
			DEBUG_OUTPUT(L"invalid nid");
		else if (!m_data_nda_table.WriteData(&m_nda_spec, sizeof(_NEURO_DATA_ALLOC_SPEC)))
			DEBUG_OUTPUT(L"failed write data");
	}
}

NeuroDataAccessManager* NeuroDataNodeAccessor::Open(NeuroStorageAllocationSystem& nsas, neuro_u32 nid)
{
	if (!m_data_nda_table.SetDataPointer(nid * sizeof(_NEURO_DATA_ALLOC_SPEC)))
	{
		DEBUG_OUTPUT(L"failed seek data node : %u", nid);
		return NULL;
	}

	if (!m_data_nda_table.ReadData(&m_nda_spec, sizeof(_NEURO_DATA_ALLOC_SPEC)))
	{
		DEBUG_OUTPUT(L"failed read data node : %u", nid);
		return NULL;
	}

	return new NeuroDataAccessManager(nsas, m_nda_spec);
}
