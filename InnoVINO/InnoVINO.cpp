// InnoVINO.cpp : Defines the initialization routines for the DLL.
//

#include "pch.h"
#include "framework.h"
#include "InnoVINO.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//

// CInnoVINOApp
//
//BEGIN_MESSAGE_MAP(CInnoVINOApp, CWinApp)
//END_MESSAGE_MAP()
//
//
//// CInnoVINOApp construction
//
//CInnoVINOApp::CInnoVINOApp()
//{
//	// TODO: add construction code here,
//	// Place all significant initialization in InitInstance
//}
//
//
//// The one and only CInnoVINOApp object
//
//CInnoVINOApp theApp;
//
//
//// CInnoVINOApp initialization
//
//BOOL CInnoVINOApp::InitInstance()
//{
//	CWinApp::InitInstance();
//
//	return TRUE;
//}


#include "COPVO.h"

void Log(LPCSTR lpMsg) {
#ifdef _INNOVINO_DEBUG_
	OutputDebugStringA(lpMsg);
#endif
}

extern "C" __declspec(dllexport) int WINAPI IVINO_Init(INT_PTR *dwServiceId) {

	Log("IVINO_Init...");

	COPVO *pOPVO = new COPVO();
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	pOPVO->Init();

	*dwServiceId = (INT_PTR)pOPVO;

	Log("IVINO_Init...done");

	return OK;
}

extern "C" __declspec(dllexport) int WINAPI IVINO_GetAvailableDevices(INT_PTR dwServiceId, AvailableDevices *pDevices) {

	Log("IVINO_GetAvailableDevices...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	pOPVO->GetAvailableDevices(pDevices);

	Log("IVINO_GetAvailableDevices...done");

	return OK;
}

extern "C" __declspec(dllexport) int WINAPI IVINO_AddEngine(INT_PTR dwServiceId, OMZ_Model *pModel) {

	Log("IVINO_AddModel...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	int nEngineId = pOPVO->AddEngine(pModel);

	Log("IVINO_AddModel...done");

	return nEngineId;
}

extern "C" __declspec(dllexport) int WINAPI IVINO_Inference(INT_PTR dwServiceId, ImageData *pData, ObjectDatas *pOutput, BOOL bAsync) {

	Log("IVINO_Inference...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	int nResult = pOPVO->Inference(pData, pOutput, bAsync);

	Log("IVINO_Inference...done");

	return nResult;
}

extern "C" __declspec(dllexport) float WINAPI IVINO_FaceRecog(INT_PTR dwServiceId, ImageData *pImage1, ImageData *pImage2, BOOL bAsync) {

	Log("IVINO_FaceRecog...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	float nResult = pOPVO->FaceRecog(pImage1, pImage2, bAsync);

	Log("IVINO_FaceRecog...done");

	return nResult;
}

//extern "C" __declspec(dllexport) int WINAPI IVINO_ConvertPtrToObjectDatas(INT_PTR dwServiceId, int type, INT_PTR pOutput, int size, INT_PTR *pDatas) {
//
//	Log("IVINO_ConvertPtrToOD_Data...");
//
//	COPVO *pOPVO = (COPVO*)dwServiceId;
//	if (pOPVO == NULL)
//		return PARAMETER_MISMATCH;
//
//	pOPVO->ConverPtrToObjectDatas(type, pOutput, size, pDatas);
//
//	Log("IVINO_ConvertPtrToOD_Data...done");
//
//	return OK;
//}
//
//extern "C" __declspec(dllexport) int WINAPI IVINO_FreeObjectDatas(INT_PTR dwServiceId, ObjectDatas pOutput) {
//
//	Log("IVINO_FreeObjectDatas...");
//
//	COPVO *pOPVO = (COPVO*)dwServiceId;
//	if (pOPVO == NULL)
//		return PARAMETER_MISMATCH;
//	
//	pOPVO->FreeObjectDatas(pOutput);
//
//	Log("IVINO_FreeObjectDatas...done");
//
//	return OK;
//}

extern "C" __declspec(dllexport) int WINAPI IVINO_Uninit(INT_PTR dwServiceId) {

	Log("IVINO_Uninit...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	pOPVO->Uninit();

	delete pOPVO;

	Log("IVINO_Uninit...done");

	return OK;
}