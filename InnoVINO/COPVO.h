#pragma once
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include "InnoVINO.h"
#include "inference_engine.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace InferenceEngine;
using namespace cv;
using namespace std;

class COPVO
{

public:
					COPVO();
					~COPVO();

	int				Init(OMZ_Model *pModel);
	int				GetAvailableDevices(AvailableDevices *pDevices);
	int				AddModel(OMZ_Model *pModel);
	int				Inference(ImageData *pImage, ObjectDatas *pOutput, BOOL bAsync);
	float			FaceRecog(ImageData *pImage1, ImageData *pImage2, BOOL bAsync);
	int				ConverPtrToObjectDatas(int type, INT_PTR pInput, int size, INT_PTR *pOutput);
	int				FreeObjectDatas(ObjectDatas pOutput);
	int				Uninit();

private:
	InferRequest	m_InferRequest;
	InputInfo::Ptr	m_InputInfo;
	DataPtr			m_OutputInfo;
	string			m_Device;
	string			m_InputName;
	string			m_OutputName;
	float			m_fXRatio;
	float			m_fYRatio;
	Device*			m_Devices;


	void			_show_model_info();
	void			_image_preprocess(Mat *pImage);
	INT_PTR			_convert_to_objects(INT_PTR pInput, int size);
	float			_cosine_similarity(const float *pfVector1, const float *pfVector2, unsigned int vector_size);
	float			_euclidean_distance(const float *pfVector1, const float *pfVector2, unsigned int vector_size);
};
