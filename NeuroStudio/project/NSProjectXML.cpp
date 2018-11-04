#include "stdafx.h"
#include "NSProjectXML.h"
#include "common.h"

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/LocalFileInputSource.hpp>
#include <xercesc/util/OutOfMemoryException.hpp>
#include <xercesc/util/PlatformUtils.hpp>

using namespace np;
using namespace np::project;
using namespace np::project::xml;

XERCES_CPP_NAMESPACE_USE

NSProjectXML::NSProjectXML()
{
	try
	{
		XMLPlatformUtils::Initialize();
	}
	catch (const XMLException& e)
	{
		DEBUG_OUTPUT(_T("XMLException[%s]"), e.getMessage());
	}

	m_document=NULL;
}

NSProjectXML::~NSProjectXML()
{
	if(m_document)
		m_document->release();
	m_document=NULL;

	try
	{
		XMLPlatformUtils::Terminate();
	}
	catch (const XMLException& e)
	{
		DEBUG_OUTPUT(_T("XMLException[%s]"), e.getMessage());
	}
}

XERCES_CPP_NAMESPACE::DOMDocument* NSProjectXML::LoadXML(const wchar_t* strXMLFilePath)
{
	if(m_document)
		m_document->release();

	if(wcslen(strXMLFilePath)==0)
		return NULL;

	XercesDOMParser parser;
	parser.setValidationScheme(XercesDOMParser::Val_Never);
	parser.setDoNamespaces(true);
	parser.setHandleMultipleImports(true);
	parser.setCreateEntityReferenceNodes(true);

    bool bRet = true;
    try
    {
		parser.parse(strXMLFilePath);
    }
    catch(const OutOfMemoryException& e)
    {
		DEBUG_OUTPUT(_T("NSProjectXML::LoadXML : OutOfMemoryException[%s]"), e.getMessage());
        bRet = false;
    }
    catch(const XMLException& e)
    {
		DEBUG_OUTPUT(_T("NSProjectXML::LoadXML : XMLException[%s]"), e.getMessage());
        bRet = false;
    }
    catch(const DOMException& e)
    {
		CString strErr;

        const unsigned int maxChars = 2047;
        XMLCh errText[maxChars + 1];
        if (DOMImplementation::loadDOMExceptionMsg(e.code, errText, maxChars))
             strErr=errText;

		DEBUG_OUTPUT(_T("NSProjectXML::LoadXML : DOMException[%s]"), strErr);
        bRet = false;
    }
    catch (...)
    {
		DEBUG_OUTPUT(_T("NSProjectXML::LoadXML : An error occurred during parsing"));
        bRet = false;
    }

	if(!bRet)
		return NULL;

	// get the DOM representation
	m_document=parser.adoptDocument();
	return m_document;
}

XERCES_CPP_NAMESPACE::DOMDocument* NSProjectXML::CreateDocument()
{
	if(m_document)
		m_document->release();
	m_document=NULL;

	DOMImplementation* impl = DOMImplementationRegistry::getDOMImplementation(XMLUni::fgXMLString);
	if(!impl)
		return NULL;

	m_document=impl->createDocument(namespace_uri::szNamespaceUri, elem::szRoot, 0);
	return m_document;
}

bool NSProjectXML::SaveXML(const wchar_t* strXMLFilePath)
{
	if(!m_document)
		return false;
#ifdef _DEBUG
	DOMNode::NodeType type=m_document->getNodeType();
#endif

	DOMImplementation* impl = DOMImplementationRegistry::getDOMImplementation(XMLUni::fgXMLString);
	if(!impl)
		return false;

    DOMLSSerializer	*writer = ((DOMImplementationLS*)impl)->createLSSerializer();
    DOMLSOutput     *theOutputDesc = ((DOMImplementationLS*)impl)->createLSOutput();
	theOutputDesc->setEncoding(_T("UTF-8"));

	bool bRet=true;
	try {
		writer->setNewLine(_T("\r\n"));

		DOMConfiguration* serializerConfig=writer->getDomConfig();
        if(serializerConfig->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
            serializerConfig->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);

        if (serializerConfig->canSetParameter(XMLUni::fgDOMWRTBOM, true))	// 요걸 해주어야 utf-8 포맷의 파일로 저장한다.
            serializerConfig->setParameter(XMLUni::fgDOMWRTBOM, true);

		LocalFileFormatTarget xmlFormTarget(strXMLFilePath);
		theOutputDesc->setByteStream(&xmlFormTarget);
		writer->write(m_document, theOutputDesc);
	}
	catch (const XMLException& e)
	{
		DEBUG_OUTPUT(_T("NSProjectXML::SaveXML : XMLException[%s]"), e.getMessage());
		bRet=false;
	}
	catch (const DOMException& e)
	{
		CString strErr;

        const unsigned int maxChars = 2047;
        XMLCh errText[maxChars + 1];
        if (DOMImplementation::loadDOMExceptionMsg(e.code, errText, maxChars))
             strErr=errText;

		DEBUG_OUTPUT(_T("NSProjectXML::SaveXML : DOMException[%s]"), strErr);
		bRet=false;
	}
	catch (...)
	{
		DEBUG_OUTPUT(_T("NSProjectXML::SaveXML : An error occurred during parsing"));
		bRet=false;
	}
    writer->release();
    theOutputDesc->release();

	return bRet;
}

unsigned int NSProjectXML::GetAttributeType(const XERCES_CPP_NAMESPACE::DOMElement& elem, LPCTSTR strAttName, const TCHAR*const* szTypeArray, unsigned int nUnknownType, unsigned int nDefaultType)
{
	return GetAttributeType(elem.getAttribute(strAttName), szTypeArray, nUnknownType, nDefaultType);
}

unsigned int NSProjectXML::GetAttributeType(LPCTSTR strType, const TCHAR*const* szTypeArray, unsigned int nUnknownType, unsigned int nDefaultType)
{
	for(unsigned int iType=0;iType<nUnknownType;iType++)
	{
		if(_tcsicmp(strType, szTypeArray[iType])==0)
			return iType;
	}
	if(nDefaultType>=0)
		return nDefaultType;
	else
		return 0;
}
