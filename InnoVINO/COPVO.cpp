#include "pch.h"
#include "COPVO.h"


COPVO::COPVO() {
	m_Devices = NULL;
}

COPVO::~COPVO() {

}


int COPVO::Init() {

	_initial_frengine("CPU");

	return OK;
}

int COPVO::GetAvailableDevices(AvailableDevices *pDevices) {
	
	//get available devices from openvino
	Core core;
	vector<string> availableDevices = core.GetAvailableDevices();

	if (m_Devices) {
		free(m_Devices);
		m_Devices = NULL;
	}
	else {
		//prepare device informaion
		m_Devices = (Device*)malloc(sizeof(Device) * availableDevices.size());
		memset(m_Devices, 0, sizeof(Device) * availableDevices.size());
	}

	int nDeviceCount = 0;
	for (int i = 0; i < availableDevices.size(); i++) {
		if (strcmp(availableDevices[i].c_str(), "GNA") != 0 && strcmp(availableDevices[i].c_str(), "GPU") != 0) {
			sprintf_s(m_Devices[nDeviceCount].szName, "%s", availableDevices[i].c_str());
			nDeviceCount++;
		}		
	}

	pDevices->nCount = nDeviceCount;
	pDevices->pDevices = (INT_PTR)m_Devices;
	
	return pDevices->nCount;
}

int COPVO::AddEngine(OMZ_Model *pModel) {

	try
	{
		//Integration Steps of OpenVINO
		//1. Initialize Core
		Core ie;
		//2. Read Model IR
		CNNNetwork cnnNetwork = ie.ReadNetwork(pModel->lpXML);
		//3. Configure Input & Output		
		InputsDataMap inputsDataMap = cnnNetwork.getInputsInfo();
		InputsDataMap::iterator input = inputsDataMap.begin();
		m_InputName = input->first;
		m_InputInfo = input->second;
		//For Multi-Device and Heterogeneous execution the supported input precision depends on the actual underlying devices. 
		//Generally, U8 is preferable as it is most ubiquitous.
		m_InputInfo->setPrecision(Precision::U8);
		OutputsDataMap outpusDataMap = cnnNetwork.getOutputsInfo();
		OutputsDataMap::iterator output = outpusDataMap.begin();
		m_OutputName = output->first;
		m_OutputInfo = output->second;
		//For Multi-Device and Heterogeneous execution the supported output precision depends on the actual underlying devices. 
		//Generally, FP32 is preferable as it is most ubiquitous.
		m_OutputInfo->setPrecision(Precision::FP32);
		//4. Load Model		
		ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, pModel->lpDevice);
		//5. Create InferRequest
		m_InferRequest = exeNetwork.CreateInferRequest();
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return 1;
	}

#ifdef _INNOVINO_DEBUG_
	_show_model_info();
#endif

	return OK;
}

int COPVO::Uninit() {

	if (mFaceFeatures.size() > 0) {
		for (int i = 0; i < mFaceFeatures.size(); i++) {
			free(mFaceFeatures[i]);
		}
		mFaceFeatures.clear();
	}
		

	if (mFaceLabels.size() > 0)
		mFaceLabels.clear();	

	//free all objects
	if (m_Devices) {
		free(m_Devices);
		m_Devices = NULL;
	}

	return OK;
}

int COPVO::Inference(ImageData *pImage, ObjectDatas *pOutput, BOOL bAsync) {

	if (pImage == NULL || pOutput == NULL)
		return PARAMETER_MISMATCH;

	try
	{
		//Scale image size that fit input size in model.
		Mat mat(pImage->uiHeight, pImage->uiWidth, CV_8UC3, (BYTE*)pImage->pData);
		_image_preprocess(&mat);

		//Do infernece and calculate the time cost.
		DWORD dwStart = GetTickCount();
		m_InferRequest.Infer();

		Blob::Ptr output_blob = m_InferRequest.GetBlob(m_OutputName);
		MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
		if (!moutput) {
			OutputDebugStringA("moutput is NULL!!!");
			return GENERAL_ERROR;
		}
		
		auto moutputHolder = moutput->rmap();
		const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
		const SizeVector outputDims = m_OutputInfo->getTensorDesc().getDims();
		const int objectSize = outputDims[1];
		const int maxProposalCount = outputDims[2];
		const SizeVector inputDims = m_InputInfo->getTensorDesc().getDims();
		const int nHeight = inputDims[2];
		const int nWidth = inputDims[3];

		m_fXRatio = (float)pImage->uiWidth / (float)nWidth;
		m_fYRatio = (float)pImage->uiHeight / (float)nHeight;
		
		vector<ObjectData> objs;
		for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
			int image_id = static_cast<int>(output[curProposal * objectSize + 0]);
			if (image_id < 0) {
				break;
			}

			ObjectData obj = { 0 };
			obj.conf = output[curProposal * objectSize + 2];
			obj.label = static_cast<int>(output[curProposal * objectSize + 1]);
			obj.x_min = static_cast<int>(output[curProposal * objectSize + 3] * nWidth);
			obj.y_min = static_cast<int>(output[curProposal * objectSize + 4] * nHeight);
			obj.x_max = static_cast<int>(output[curProposal * objectSize + 5] * nWidth);
			obj.y_max = static_cast<int>(output[curProposal * objectSize + 6] * nHeight);

			//rescale the object position.
			obj.x_min *= m_fXRatio;
			obj.x_max *= m_fXRatio;
			obj.y_min *= m_fYRatio;
			obj.y_max *= m_fYRatio;

			objs.push_back(obj);
		}

		pOutput->nCount = objs.size();
		if (pOutput->nCount > 0) {
			ObjectData* datas = (ObjectData*)malloc(sizeof(ObjectData) * pOutput->nCount);
			if (datas) {
				memset(datas, 0, sizeof(ObjectData) * pOutput->nCount);
				for (int n = 0; n < pOutput->nCount; n++) {
					datas[n] = objs[n];
				}
				pOutput->pObjects = (INT_PTR)datas;
			}
			objs.clear();
		}

		//If inference work than return with the number of bbox.
		return pOutput->nCount;
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return GENERAL_ERROR;
	}
}

float COPVO::FaceRecogEx(ImageData *pImage, ObjectData *pOutput) {

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage->uiHeight, pImage->uiWidth, CV_8UC3, (BYTE*)pImage->pData);
		_facerecognition_preprocess(&img1);

		//Do infernece and calculate the time cost.
		DWORD dwStart = GetTickCount();
		mFaceRecognition_InferRequest.Infer();

		Blob::Ptr output_blob = mFaceRecognition_InferRequest.GetBlob(mFaceRecognition_OutputName);
		MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
		if (!moutput) {
			OutputDebugStringA("moutput1 is NULL!!!");
			return GENERAL_ERROR;
		}

		auto moutputHolder = moutput->rmap();
		const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
		float fFace[513] = { 0.0f };
		memcpy(fFace, output, 512);
		
		float highestValue = 0.0f;
		int	  highestIndex = 0;

		for (int i = 0 ; i < mFaceFeatures.size() - 1 ; i++ ) {
			float result = _cosine_similarity(fFace, mFaceFeatures[i], 512);
			if (result > highestValue) {
				highestValue = result;
				highestIndex = i;
			}
		}

		if (pOutput!= NULL && highestValue != 0.0f) {
			pOutput->conf = highestValue;
			pOutput->label = stoi(mFaceLabels[highestIndex]);
		}

		return highestValue;
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return GENERAL_ERROR;
	}

	return 0.0f;
}

float COPVO::FaceRecog(ImageData *pImage1, ImageData *pImage2, BOOL bAsync) {

	float fconf = 0.0f;

	if (pImage1 == NULL || pImage2 == NULL)
		return PARAMETER_MISMATCH;

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage1->uiHeight, pImage1->uiWidth, CV_8UC3, (BYTE*)pImage1->pData);		
		_facerecognition_preprocess(&img1);

		//Do infernece and calculate the time cost.
		DWORD dwStart = GetTickCount();
		mFaceRecognition_InferRequest.Infer();

		Blob::Ptr output_blob = mFaceRecognition_InferRequest.GetBlob(mFaceRecognition_OutputName);
		MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
		if (!moutput) {
			OutputDebugStringA("moutput1 is NULL!!!");
			return GENERAL_ERROR;
		}

		auto moutputHolder = moutput->rmap();
		const float *output1 = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
		float fFace1[513] = { 0 };
		memcpy(fFace1, output1, 512);

		//Scale image size that fit input size in model.
		Mat img2(pImage2->uiHeight, pImage2->uiWidth, CV_8UC3, (BYTE*)pImage2->pData);
		_facerecognition_preprocess(&img2);

		//Do infernece and calculate the time cost.
		dwStart = GetTickCount();
		mFaceRecognition_InferRequest.Infer();

		Blob::Ptr output_blob1 = mFaceRecognition_InferRequest.GetBlob(mFaceRecognition_OutputName);
		MemoryBlob::CPtr moutput1 = as<MemoryBlob>(output_blob1);
		if (!moutput1) {
			OutputDebugStringA("moutput2 is NULL!!!");
			return GENERAL_ERROR;
		}

		auto moutputHolder1 = moutput1->rmap();
		const float *output2 = moutputHolder1.as<const PrecisionTrait<Precision::FP32>::value_type *>();
		float fFace2[513] = { 0 };
		memcpy(fFace2, output2, 512);

		fconf = _cosine_similarity(fFace1, fFace2, 512);

		return fconf;
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return GENERAL_ERROR;
	}
}

void COPVO::_show_model_info() {

	char szMsg[MAX_PATH] = { 0 };

	if (m_InputInfo) {
		const SizeVector inputDims = m_InputInfo->getTensorDesc().getDims();
		sprintf_s(szMsg, "Input [BxCxHxW] : [%dx%dx%dx%d]", inputDims[0], inputDims[1], inputDims[2], inputDims[3]);
		OutputDebugStringA(szMsg);
		sprintf_s(szMsg, "Precesioin : %d", (int)m_InputInfo->getPrecision());
		OutputDebugStringA(szMsg);
	}

	if (!m_InputName.empty()) {
		OutputDebugStringA(m_InputName.c_str());
	}
	
	if (m_OutputInfo) {
		const SizeVector outputDims = m_OutputInfo->getTensorDesc().getDims();
		sprintf_s(szMsg, "Output [x, x, x, x] : [%d, %d, %d, %d]", outputDims[0], outputDims[1], outputDims[2], outputDims[3]);
		OutputDebugStringA(szMsg);

		sprintf_s(szMsg, "Precesioin : %d", (int)m_OutputInfo->getPrecision());
		OutputDebugStringA(szMsg);
	}

	if (!m_OutputName.empty()) {
		OutputDebugStringA(m_OutputName.c_str());
	}
}

void COPVO::_image_preprocess(Mat *pImage) {

	if (m_InputInfo == NULL)
		return;

	const SizeVector inputDims = m_InputInfo->getTensorDesc().getDims();
	const int nChanelNum = inputDims[1];
	const int nHeight = inputDims[2];
	const int nWidth = inputDims[3];
		
	Mat resized;
	resize(*pImage, resized, cv::Size(nWidth, nHeight));

	Blob::Ptr blob = m_InferRequest.GetBlob(m_InputName);
	unsigned char* ptr = (unsigned char*)blob->buffer();
	for (int c = 0; c < nChanelNum; ++c) {
		for (int y = 0; y < nHeight; ++y) {
			for (int x = 0; x < nWidth; ++x) {
				*(ptr++) = resized.at<Vec3b>(y, x)[c];
			}
		}
	}
}

float COPVO::_cosine_similarity(const float *pfVector1, const float *pfVector2, unsigned int vector_size) {

	float dot = 0.0, denom_a = 0.0, denom_b = 0.0;

	for (unsigned int i = 0u; i < vector_size; ++i) {
		dot += pfVector1[i] * pfVector2[i];
		denom_a += pfVector1[i] * pfVector1[i];
		denom_b += pfVector2[i] * pfVector2[i];
	}

	return dot / (sqrt(denom_a) * sqrt(denom_b));
}

float COPVO::_euclidean_distance(const float *pfVector1, const float *pfVector2, unsigned int vector_size) {

	float square = 0.0f, result = 0.0f;

	for (int i = 0; i < vector_size; i++) {
		square += (pfVector1[i] - pfVector2[i]) * (pfVector1[i] - pfVector2[i]);
	}

	return sqrt(square);
}

int COPVO::_initial_frengine(LPCSTR lpDevice) {

	try
	{
		//Face Detection Engine initial
		//1. Initialize Core
		Core ie;
		//2. Read Model IR
		char szPath[MAX_PATH] = { 0 };
		GetModuleFileNameA(AfxGetInstanceHandle(), szPath, MAX_PATH);
		*(strrchr(szPath, '\\') + 1) = 0;
		strcat(szPath, "face-detection-0102.xml");
		CNNNetwork cnnNetwork = ie.ReadNetwork(szPath);
		//3. Configure Input & Output		
		InputsDataMap inputsDataMap = cnnNetwork.getInputsInfo();
		InputsDataMap::iterator input = inputsDataMap.begin();
		mFaceDetect_InputName = input->first;
		mFaceDetect_InputInfo = input->second;
		//For Multi-Device and Heterogeneous execution the supported input precision depends on the actual underlying devices. 
		//Generally, U8 is preferable as it is most ubiquitous.
		mFaceDetect_InputInfo->setPrecision(Precision::U8);
		OutputsDataMap outpusDataMap = cnnNetwork.getOutputsInfo();
		OutputsDataMap::iterator output = outpusDataMap.begin();
		mFaceDetect_OutputName = output->first;
		mFaceDetect_OutputInfo = output->second;
		//For Multi-Device and Heterogeneous execution the supported output precision depends on the actual underlying devices. 
		//Generally, FP32 is preferable as it is most ubiquitous.
		mFaceDetect_OutputInfo->setPrecision(Precision::FP32);
		//4. Load Model		
		ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, lpDevice);
		//5. Create InferRequest
		mFaceDetect_InferRequest = exeNetwork.CreateInferRequest();

		//Face Recognition Engine initial
		//1. Initialize Core
		Core fr_ie;
		//2. Read Model IR
		GetModuleFileNameA(AfxGetInstanceHandle(), szPath, MAX_PATH);
		*(strrchr(szPath, '\\') + 1) = 0;
		strcat(szPath, "Sphereface.xml");
		CNNNetwork fr_cnnNetwork = fr_ie.ReadNetwork(szPath);
		//3. Configure Input & Output		
		InputsDataMap fr_inputsDataMap = fr_cnnNetwork.getInputsInfo();
		InputsDataMap::iterator fr_input = fr_inputsDataMap.begin();
		mFaceRecognition_InputName = fr_input->first;
		mFaceRecognition_InputInfo = fr_input->second;
		//For Multi-Device and Heterogeneous execution the supported input precision depends on the actual underlying devices. 
		//Generally, U8 is preferable as it is most ubiquitous.
		mFaceRecognition_InputInfo->setPrecision(Precision::U8);
		OutputsDataMap fr_outpusDataMap = fr_cnnNetwork.getOutputsInfo();
		OutputsDataMap::iterator fr_output = fr_outpusDataMap.begin();
		mFaceRecognition_OutputName = fr_output->first;
		mFaceRecognition_OutputInfo = fr_output->second;
		//For Multi-Device and Heterogeneous execution the supported output precision depends on the actual underlying devices. 
		//Generally, FP32 is preferable as it is most ubiquitous.
		mFaceRecognition_OutputInfo->setPrecision(Precision::FP32);
		//4. Load Model		
		ExecutableNetwork fr_exeNetwork = fr_ie.LoadNetwork(fr_cnnNetwork, lpDevice);
		//5. Create InferRequest
		mFaceRecognition_InferRequest = fr_exeNetwork.CreateInferRequest();
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return 1;
	}
}

void COPVO::_facedetection_preprocess(Mat *pImage) {

	if (mFaceDetect_InputInfo == NULL)
		return;

	const SizeVector inputDims = mFaceDetect_InputInfo->getTensorDesc().getDims();
	const int nChanelNum = inputDims[1];
	const int nHeight = inputDims[2];
	const int nWidth = inputDims[3];

	Mat resized;
	resize(*pImage, resized, cv::Size(nWidth, nHeight));

	Blob::Ptr blob = mFaceDetect_InferRequest.GetBlob(mFaceDetect_InputName);
	unsigned char* ptr = (unsigned char*)blob->buffer();
	for (int c = 0; c < nChanelNum; ++c) {
		for (int y = 0; y < nHeight; ++y) {
			for (int x = 0; x < nWidth; ++x) {
				*(ptr++) = resized.at<Vec3b>(y, x)[c];
			}
		}
	}
}

void COPVO::_facerecognition_preprocess(Mat *pImage) {
	if (mFaceRecognition_InputInfo == NULL)
		return;

	const SizeVector inputDims = mFaceRecognition_InputInfo->getTensorDesc().getDims();
	const int nChanelNum = inputDims[1];
	const int nHeight = inputDims[2];
	const int nWidth = inputDims[3];

	Mat resized;
	resize(*pImage, resized, cv::Size(nWidth, nHeight));
	
	Blob::Ptr blob = mFaceRecognition_InferRequest.GetBlob(mFaceRecognition_InputName);
	unsigned char* ptr = (unsigned char*)blob->buffer();
	for (int c = 0; c < nChanelNum; ++c) {
		for (int y = 0; y < nHeight; ++y) {
			for (int x = 0; x < nWidth; ++x) {
				*(ptr++) = resized.at<Vec3b>(y, x)[c];
			}
		}
	}
}

int	COPVO::AddFace(ImageData *pImage, LPCSTR lpLabel) {

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage->uiHeight, pImage->uiWidth, CV_8UC3, (BYTE*)pImage->pData);
		_facerecognition_preprocess(&img1);

		//Do infernece and calculate the time cost.
		DWORD dwStart = GetTickCount();
		mFaceRecognition_InferRequest.Infer();

		Blob::Ptr output_blob = mFaceRecognition_InferRequest.GetBlob(mFaceRecognition_OutputName);
		MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
		if (!moutput) {
			OutputDebugStringA("moutput1 is NULL!!!");
			return GENERAL_ERROR;
		}

		auto moutputHolder = moutput->rmap();
		const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
		/*float fFace[513] = { 0 };*/
		float* fFace = (float*)malloc(sizeof(float) * 513);
		memset(fFace, 0.0f, sizeof(float) * 513);
		memcpy(fFace, output, 512);

		mFaceFeatures.push_back(fFace);
		mFaceLabels.push_back(lpLabel);
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return GENERAL_ERROR;
	}

	return OK;
}

