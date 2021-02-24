// InnoVINO.h : main header file for the InnoVINO DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

#ifndef __INNOVINO__
#define __INNOVINO__

#define USER_ID			5289
#define INNOVINO_OUTPUT WM_USER + USER_ID

enum ModelType
{
	OBJECT_DETECTION_GENERAL,
	SPHEREFACE,
};

enum DeviceType
{
	CPU,
	GNA,
	GPU,
	MYRIAD,
};

#pragma pack(1)
typedef struct tagOMZ_Model {
	LPCSTR		lpXML;
	LPCSTR		lpBIN;
}OMZ_Model;

typedef struct tagDevice {
	INT32		nType;
	char		szName[10];
}Device;

typedef struct tagAvailableDevices {
	int			nCount;
	INT_PTR		pDevices;
}AvailableDevices;

typedef struct tagImageData {
	UINT16		uiWidth;
	UINT16		uiHeight;
	UINT32		uiSize;
	INT_PTR		pData;
}ImageData;

//Object data
typedef struct tagObjectData {
	float		conf;
	INT32		label;
	UINT16		x_min;
	UINT16		y_min;
	UINT16		x_max;
	UINT16		y_max;
}ObjectData;

//Object detection datas
typedef struct tagObjectDatas {
	int			nCount;
	INT_PTR		pObjects;
}ObjectDatas;
#pragma pack()
#endif