// InnoVINO.h : main header file for the InnoVINO DLL

#define _LINUX_API_ 1

#pragma once

#ifdef _WINDOWS_API_

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

#endif

#ifdef _LINUX_API_

#endif


#ifndef __INNOVINO__
#define __INNOVINO__

#ifdef _WINDOWS_API_
#define USER_ID			5289
#define INNOVINO_OUTPUT WM_USER + USER_ID
#endif

#ifdef _LINUX_API_

#endif

enum ModelType
{
	OBJECT_DETECTION_GENERAL,
	SPHEREFACE,
};

#pragma pack(1)
typedef struct tagOMZ_Model {

#ifdef _WINDOWS_API_
	LPCSTR		lpXML;
	LPCSTR		lpBIN;
#endif
#ifdef _LINUX_API_
	char*		lpXML;
	char*		lpBIN;
#endif

}OMZ_Model;

typedef struct tagImageData {
#ifdef _WINDOWS_API_
	UINT16		uiWidth;
	UINT16		uiHeight;
	UINT32		uiSize;
	INT_PTR		pData;
#endif
#ifdef _LINUX_API_
	unsigned short	uiWidth;
	unsigned short	uiHeight;
	unsigned int	uiSize;
	void*			pData;
#endif

}ImageData;

//Object data
typedef struct tagObjectData {
#ifdef _WINDOWS_API_
	float		conf;
	INT32		label;
	UINT16		x_min;
	UINT16		y_min;
	UINT16		x_max;
	UINT16		y_max;
#endif
#ifdef _LINUX_API_
	float			conf;
	unsigned int	label;
	unsigned short	x_min;
	unsigned short	y_min;
	unsigned short	x_max;
	unsigned short	y_max;
#endif
}ObjectData;

//Object detection datas
typedef struct tagObjectDatas {
#ifdef _WINDOWS_API_
	int			nCount;
	INT_PTR		pObjects;
#endif
#ifdef _LINUX_API_
	int			nCount;
	void*		pObjects;
#endif
}ObjectDatas;
#pragma pack()

typedef int (*IVINO_INIT)(unsigned long *pulServiceId, OMZ_Model *pModel);
typedef	int (*IVINO_ADDMODEL)(unsigned long ulServiceId, OMZ_Model *pModel);
typedef	ObjectDatas* (*IVINO_INFERENCE)(unsigned long ulServiceId, ImageData *pData, bool bAsync);
typedef	float (*IVINO_FACERECOG)(unsigned long ulServiceId, ImageData *pImage1, ImageData *pImage2, bool bAsync);
typedef int (*IVINO_FREEOBJECTDATAS)(unsigned long dwServiceId, ObjectDatas *pOutput);
typedef int (*IVINO_UNINIT)(unsigned long ulServiceId);

// extern "C"{
// 	int IVINO_Init(unsigned long *pulServiceId, OMZ_Model *pModel);
// 	int IVINO_AddModel(unsigned long ulServiceId, OMZ_Model *pModel);
// 	int IVINO_Inference(unsigned long ulServiceId, ImageData *pData, ObjectDatas *pOutput, bool bAsync);
// 	float IVINO_FaceRecog(unsigned long ulServiceId, ImageData *pImage1, ImageData *pImage2, bool bAsync);
// 	int IVINO_FreeObjectDatas(unsigned long dwServiceId, ObjectDatas *pOutput);
// 	int IVINO_Uninit(unsigned long ulServiceId);
// }

#endif



