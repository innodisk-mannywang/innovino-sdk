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

	int				Init();
	int				GetAvailableDevices(AvailableDevices *pDevices);
	int				AddEngine(OMZ_Model *pModel);
	int				Inference(ImageData *pImage, ObjectDatas *pOutput, BOOL bAsync);
	int				AddFace(ImageData *pImage, LPCSTR lpLabel);
	float			FaceRecogEx(ImageData *pImage, ObjectData *pOutput);
	float			FaceRecog(ImageData *pImage1, ImageData *pImage2, BOOL bAsync);
	int				Uninit();

private:
	InferRequest	m_InferRequest;
	InputInfo::Ptr	m_InputInfo;
	DataPtr			m_OutputInfo;
	string			m_InputName;
	string			m_OutputName;
	float			m_fXRatio;
	float			m_fYRatio;
	Device*			m_Devices;

	//For Face Recognition
	InferRequest	mFaceDetect_InferRequest, mFaceRecognition_InferRequest;
	InputInfo::Ptr	mFaceDetect_InputInfo, mFaceRecognition_InputInfo;
	DataPtr			mFaceDetect_OutputInfo, mFaceRecognition_OutputInfo;
	string			mFaceDetect_InputName, mFaceRecognition_InputName;
	string			mFaceDetect_OutputName, mFaceRecognition_OutputName;
	vector<float*>	mFaceFeatures;
	vector<string>	mFaceLabels;
	vector<Mat>		mFaceMats;

	void			_show_model_info();
	void			_image_preprocess(Mat *pImage);
	INT_PTR			_convert_to_objects(INT_PTR pInput, int size);
	float			_cosine_similarity(const float *pfVector1, const float *pfVector2, unsigned int vector_size);
	float			_euclidean_distance(const float *pfVector1, const float *pfVector2, unsigned int vector_size);

	//For Face Recognition
	int				_initial_frengine(LPCSTR lpDevice);
	void			_facedetection_preprocess(Mat *pImage);
	void			_facerecognition_preprocess(Mat *pImage);
	int				_write_features();
	int				_read_features();
};
