#include "ImageListBox.h"

#include "NeuroCommon/JNM_Common.h"

namespace ahnn
{
	namespace windows
	{
		class CImageFileListBox : public CImageListBox
		{
		public:
			CImageFileListBox();
			virtual	~CImageFileListBox();

			bool AddFile(const wchar_t* file_path);

		protected:
			virtual void DeleteItemData(int nIndex);

		private:
			CImageList m_imgList;

			SIZE m_cxImgSize;

		protected:
			DECLARE_MESSAGE_MAP()
			afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		};
	}
}
