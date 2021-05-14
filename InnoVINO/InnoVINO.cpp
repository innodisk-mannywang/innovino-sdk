// InnoVINO.cpp : Defines the initialization routines for the DLL.
//

#include "InnoVINO.h"
#include "COPVO.h"

void Log(const char *lpMsg) {
#ifdef _WINDOWS_API_
	OutputDebugStringA(lpMsg);
#endif
#ifdef _LINUX_API_
	char szMsg[256] = {0};
	sprintf(szMsg, "%s\n", lpMsg);
	printf(szMsg);
#endif
}

#ifdef _WINDOWS_API_
#include "pch.h"
#include "framework.h"
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
#endif

#ifdef _LINUX_API_
#define INT_PTR	unsigned long;
#endif

#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) int WINAPI IVINO_Init(INT_PTR *dwServiceId, OMZ_Model *pModel) 
#endif
#ifdef _LINUX_API_
extern "C" int IVINO_Init(unsigned long *dwServiceId, OMZ_Model *pModel)
#endif
{
	Log("IVINO_Init...");

	COPVO *pOPVO = new COPVO();	
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	Log(pModel->lpBIN);
	Log(pModel->lpXML);

	pOPVO->Init(pModel);

#ifdef _WINDOWS_API_
	*dwServiceId = (INT_PTR)pOPVO;
#endif
#ifdef _LINUX_API_
	*dwServiceId = (unsigned long)pOPVO;
#endif

	Log("IVINO_Init...done");

	return OK;
}

#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) int WINAPI IVINO_AddModel(INT_PTR dwServiceId, OMZ_Model *pModel) 
#endif
#ifdef _LINUX_API_
extern "C" int IVINO_AddModel(unsigned long dwServiceId, OMZ_Model *pModel)
#endif
{
	Log("IVINO_AddModel...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;



	Log("IVINO_AddModel...done");

	return OK;
}

#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) int WINAPI IVINO_Inference(INT_PTR dwServiceId, ImageData *pData, ObjectDatas *pOutput, BOOL bAsync) 
#endif
#ifdef _LINUX_API_
extern "C" int IVINO_Inference(unsigned long dwServiceId, ImageData *pData, ObjectDatas *pOutput, bool bAsync)
#endif
{
	Log("IVINO_Inference...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	int nResult = pOPVO->Inference(pData, pOutput, bAsync);
	//int nResult = 0;
	Log("IVINO_Inference...done");

	return nResult;
}


#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) float WINAPI IVINO_FaceRecog(INT_PTR dwServiceId, ImageData *pImage1, ImageData *pImage2, BOOL bAsync) 
#endif
#ifdef _LINUX_API_
extern "C" float IVINO_FaceRecog(unsigned long dwServiceId, ImageData *pImage1, ImageData *pImage2, bool bAsync)
#endif
{
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


#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) int WINAPI IVINO_FreeObjectDatas(INT_PTR dwServiceId, ObjectDatas* pOutput) 
#endif
#ifdef _LINUX_API_
extern "C" int IVINO_FreeObjectDatas(unsigned long dwServiceId, ObjectDatas* pOutput)
#endif
{
	Log("IVINO_FreeObjectDatas...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;
	
	pOPVO->FreeObjectDatas(pOutput);

	Log("IVINO_FreeObjectDatas...done");

	return OK;
}

#ifdef _WINDOWS_API_
extern "C" __declspec(dllexport) int WINAPI IVINO_Uninit(INT_PTR dwServiceId)
#endif
#ifdef _LINUX_API_
extern "C" int IVINO_Uninit(unsigned long dwServiceId)
#endif
{
	Log("IVINO_Uninit...");

	COPVO *pOPVO = (COPVO*)dwServiceId;
	if (pOPVO == NULL)
		return PARAMETER_MISMATCH;

	pOPVO->Uninit();

	delete pOPVO;

	Log("IVINO_Uninit...done");

	return OK;
}