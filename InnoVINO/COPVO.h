#pragma once
#pragma warning(disable : 4996) //_CRT_SECURE_NO_WARNINGS

#include "InnoVINO.h"
#include "inference_engine.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <stdio.h>

using namespace InferenceEngine;
using namespace cv;
using namespace std;

class COPVO
{
public:
					COPVO();
					~COPVO();

	int				Init(OMZ_Model *pModel);
	int				AddModel(OMZ_Model *pModel);
	int				Inference(ImageData *pImage, ObjectDatas *pOutput, bool bAsync);
	int				Inference(ImageData *pImage, bool bAsync);
	float			FaceRecog(ImageData *pImage1, ImageData *pImage2, bool bAsync);
	// int				ConverPtrToObjectDatas(int type, void *pInput, int size, void *pOutput);
	int				FreeObjectDatas(ObjectDatas *pOutput);
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

	void			_show_model_info();
	void			_image_preprocess(Mat *pImage);
	void*			_convert_to_objects(void *pInput, int size);
	float			_cosine_similarity(const float *pfVector1, const float *pfVector2, unsigned int vector_size);
	float			_euclidean_distance(const float *pfVector1, const float *pfVector2, unsigned int vector_size);
};
