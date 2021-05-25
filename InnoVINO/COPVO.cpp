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
		
		int nType = OTHER;

		if (strcmp(availableDevices[i].c_str(), "CPU") == 0)
		{
			nType = CPU;
		}
		else if (strcmp(availableDevices[i].c_str(), "GPU") == 0) 
		{
			nType = GPU;
		}
		else if (strcmp(availableDevices[i].c_str(), "MYRIAD") == 0)
		{
			nType = MYRIAD;
		}
		else if (strcmp(availableDevices[i].c_str(), "GNA") == 0)
		{
			nType = GNA;
		}

		m_Devices[nDeviceCount].nType = CPU;
		sprintf_s(m_Devices[nDeviceCount].szName, "%s", availableDevices[i].c_str());
		nDeviceCount++;
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

void COPVO::FreeObjects(ObjectDatas *pOutput)
{
	if (pOutput && pOutput->pObjects) {
		free((ObjectDatas*)pOutput->pObjects);
		pOutput->pObjects = NULL;
	}
}

float COPVO::FaceRecogEx(ImageData *pImage, ObjectData *pOutput) {

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage->uiHeight, pImage->uiWidth, CV_8UC3, (BYTE*)pImage->pData);
		//Face Alignment
		_facealignment_preprocess(&img1);
		//Face detection
		_facedetection_preprocess(&img1);
		//Get face feature
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
		/*const int nFaceFeatures = 512;*/
		const int nFaceFeatures = 256;
		float fFace[nFaceFeatures + 1] = { 0.0f };
		memcpy(fFace, output, nFaceFeatures);
		
		float highestValue = 0.0f;
		int	  highestIndex = 0;

		for (int i = 0 ; i < mFaceFeatures.size() - 1 ; i++ ) {
			float result = _cosine_similarity(fFace, mFaceFeatures[i], nFaceFeatures);
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
		//Face Alignment
		_facealignment_preprocess(&img1);
		//Get face features
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
		const int nFaceFeatures = 256;
		float fFace1[nFaceFeatures + 1] = { 0 };
		memcpy(fFace1, output1, nFaceFeatures);

		//Scale image size that fit input size in model.
		Mat img2(pImage2->uiHeight, pImage2->uiWidth, CV_8UC3, (BYTE*)pImage2->pData);
		//Face Alignment
		_facealignment_preprocess(&img2);
		//Get face features
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
		float fFace2[nFaceFeatures + 1] = { 0 };
		memcpy(fFace2, output2, nFaceFeatures);

		fconf = _cosine_similarity(fFace1, fFace2, nFaceFeatures);

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

	*pImage = resized;
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
		OutputDebugStringA(szPath);
		CNNNetwork cnnNetwork = ie.ReadNetwork(szPath);
		OutputDebugStringA("2. Read Model IR");
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
		OutputDebugStringA("3. Configure Input & Output");
		//4. Load Model		
		ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, lpDevice);
		OutputDebugStringA("4. Load Model");
		//5. Create InferRequest
		mFaceDetect_InferRequest = exeNetwork.CreateInferRequest();

		OutputDebugStringA("Face Detection Engine initial finish!");

		//Face Recognition Engine initial
		//1. Initialize Core
		Core fr_ie;
		//2. Read Model IR
		GetModuleFileNameA(AfxGetInstanceHandle(), szPath, MAX_PATH);
		*(strrchr(szPath, '\\') + 1) = 0;
		strcat(szPath, "face-reidentification-retail-0095.xml");
		//strcat(szPath, "Sphereface.xml");
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

		OutputDebugStringA("Face Recognition Engine initial finish!");

		//Face Alignment Engine initial
		//1. Initialize Core
		Core fa_ie;
		//2. Read Model IR
		GetModuleFileNameA(AfxGetInstanceHandle(), szPath, MAX_PATH);
		*(strrchr(szPath, '\\') + 1) = 0;
		strcat(szPath, "landmarks-regression-retail-0009.xml");
		OutputDebugStringA(szPath);
		CNNNetwork fa_cnnNetwork = fa_ie.ReadNetwork(szPath);
		//3. Configure Input & Output
		InputsDataMap fa_inputsDataMap = fa_cnnNetwork.getInputsInfo();
		InputsDataMap::iterator fa_input = fa_inputsDataMap.begin();
		mFaceAlignment_InputName = fa_input->first;
		mFaceAlignment_InputInfo = fa_input->second;
		//For Multi-Device and Heterogeneous execution the supported input precision depends on the actual underlying devices.
		//Generally, U8 is preferable as it is most ubiquitous.
		mFaceAlignment_InputInfo->setPrecision(Precision::U8);
		OutputsDataMap fa_outpusDataMap = fa_cnnNetwork.getOutputsInfo();
		OutputsDataMap::iterator fa_output = fa_outpusDataMap.begin();
		mFaceAlignment_OutputName = fa_output->first;
		mFaceAlignment_OutputInfo = fa_output->second;
		//For Multi-Device and Heterogeneous execution the supported output precision depends on the actual underlying devices.
		//Generally, FP32 is preferable as it is most ubiquitous.
		mFaceAlignment_OutputInfo->setPrecision(Precision::FP32);
		//4. Load Model
		ExecutableNetwork fa_exeNetwork = fa_ie.LoadNetwork(fa_cnnNetwork, lpDevice);
		//5. Create InferRequest
		mFaceAlignment_InferRequest = fa_exeNetwork.CreateInferRequest();

		OutputDebugStringA("Face Alignment Engine initial finish!");
	}
	catch (const std::exception& e)
	{
		OutputDebugStringA(e.what());
		return 1;
	}
}

void COPVO::_facedetection_preprocess(Mat *pImage) {

	if (mFaceDetect_InputInfo == NULL || pImage == NULL)
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

	mFaceDetect_InferRequest.Infer();

	Blob::Ptr output_blob = mFaceDetect_InferRequest.GetBlob(mFaceDetect_OutputName);
	MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
	if (!moutput) {
		OutputDebugStringA("moutput is NULL!!!");
		return ;
	}

	auto moutputHolder = moutput->rmap();
	const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
	const SizeVector outputDims = mFaceDetect_OutputInfo->getTensorDesc().getDims();
	const int objectSize = outputDims[1];	
	
	m_fXRatio = (float)pImage->cols / (float)resized.cols;
	m_fYRatio = (float)pImage->rows / (float)resized.rows;

	vector<ObjectData> objs;
	for (int curProposal = 0; curProposal < 1; curProposal++) {
		ObjectData obj = { 0 };
		obj.conf = output[curProposal * objectSize + 2];
		obj.label = static_cast<int>(output[curProposal * objectSize + 1]);
		//sprintf(obj.label, "%d", static_cast<int>(output[curProposal * objectSize + 1]));
		obj.x_min = static_cast<int>(output[curProposal * objectSize + 3] * resized.cols) ;
		obj.y_min = static_cast<int>(output[curProposal * objectSize + 4] * resized.rows);
		obj.x_max = static_cast<int>(output[curProposal * objectSize + 5] * resized.cols);
		obj.y_max = static_cast<int>(output[curProposal * objectSize + 6] * resized.rows);

		//rescale the object position.
		/*obj.x_min *= m_fXRatio;
		obj.x_max *= m_fXRatio;
		obj.y_min *= m_fYRatio;
		obj.y_max *= m_fYRatio;*/

		/*char szMsg[MAX_PATH] = { 0 };
		sprintf(szMsg, "x : %d, y : %d, width : %d, height : %d", obj.x_min, obj.y_min, (obj.x_max - obj.x_min + 1), (obj.y_max - obj.y_min + 1));
		OutputDebugStringA(szMsg);*/

		/*Mat m_roi = *pImage(cv::Rect(250, 300, 100, 100));*/
		/*Mat roi = *pImage;
		roi(cv::Rect(obj.x_min, obj.y_min, (obj.x_max - obj.x_min + 1), (obj.y_max - obj.y_min + 1)));*/
		//imshow("roi", roi);
		Mat roi = resized(cv::Rect(obj.x_min, obj.y_min, (obj.x_max - obj.x_min + 1), (obj.y_max - obj.y_min + 1)));
		//imshow("roi", roi);
		*pImage = roi;
	}
}

void COPVO::_facerecognition_preprocess(Mat *pImage) {
	if (mFaceRecognition_InputInfo == NULL || pImage == NULL)
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

void COPVO::_facealignment_preprocess(Mat *pImage) {

	if (mFaceAlignment_InputInfo == NULL || pImage == NULL)
		return;

	const SizeVector inputDims = mFaceAlignment_InputInfo->getTensorDesc().getDims();
	const int nChanelNum = inputDims[1];
	const int nHeight = inputDims[2];
	const int nWidth = inputDims[3];

	Mat resized;
	resize(*pImage, resized, cv::Size(nWidth, nHeight));

	Blob::Ptr blob = mFaceAlignment_InferRequest.GetBlob(mFaceAlignment_InputName);
	unsigned char* ptr = (unsigned char*)blob->buffer();
	for (int c = 0; c < nChanelNum; ++c) {
		for (int y = 0; y < nHeight; ++y) {
			for (int x = 0; x < nWidth; ++x) {
				*(ptr++) = resized.at<Vec3b>(y, x)[c];
			}
		}
	}

	//Get landmark
	mFaceAlignment_InferRequest.Infer();

	Blob::Ptr output_blob = mFaceAlignment_InferRequest.GetBlob(mFaceAlignment_OutputName);
	MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
	if (!moutput) {
		OutputDebugStringA("moutput is NULL!!!");
		return;
	}

	auto moutputHolder = moutput->rmap();
	const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
	float landmark[11] = { 0.0f };
	memcpy(landmark, output, 10 * sizeof(float));

	float angler = _get_angle_2points(landmark[0] * resized.cols, landmark[1] * resized.rows, landmark[2] * resized.cols, landmark[3] * resized.rows);

	Point2f pt((landmark[0] * pImage->cols + landmark[2] * pImage->cols) / 2, (landmark[1] * pImage->rows + landmark[3] * pImage->rows) / 2);          //point from where to rotate		
	Mat r = getRotationMatrix2D(pt, angler, 1.0);      //Mat object for storing after rotation

	Mat dst;      //Mat object for output image file
	warpAffine(*pImage, dst, r, Size(pImage->cols, pImage->rows));  ///applie an affine transforation to image.

	*pImage = dst;
}

float COPVO::_get_angle_2points(int p1x, int p1y, int p2x, int p2y) {
	int deltaY = p2y - p1y;
	int deltaX = p2x - p1x;
	return atan2(deltaY, deltaX) * 180 / 3.141;
}

int	COPVO::AddFace(ImageData *pImage, LPCSTR lpLabel) {

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage->uiHeight, pImage->uiWidth, CV_8UC3, (BYTE*)pImage->pData);
		//Face Alignment
		_facealignment_preprocess(&img1);
		//Face detection
		_facedetection_preprocess(&img1);
		//Face recognition
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
		/*const int nFaceFeatures = 512;*/
		const int nFaceFeatures = 256;
		float* fFace = (float*)malloc(sizeof(float) * (nFaceFeatures + 1));
		memset(fFace, 0.0f, sizeof(float) * (nFaceFeatures + 1));
		memcpy(fFace, output, nFaceFeatures);

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
