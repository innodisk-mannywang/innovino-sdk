#include "pch.h"
#include "COPVO.h"


COPVO::COPVO() {
	m_Device = "CPU";
	//m_Device = "MYRIAD";
}

COPVO::~COPVO() {

}


int COPVO::Init(OMZ_Model *pModel) {

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
		ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, m_Device);		
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

int COPVO::AddModel(OMZ_Model *pModel) {

	return OK;
}

int COPVO::Uninit() {


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
		
		char szMsg[MAX_PATH] = { 0 };
		/*sprintf_s(szMsg, "Infer time cost : %u", GetTickCount() - dwStart);
		OutputDebugStringA(szMsg);*/
				
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

float COPVO::FaceRecog(ImageData *pImage1, ImageData *pImage2, BOOL bAsync) {

	float fconf = 0.0f;

	if (pImage1 == NULL || pImage2 == NULL)
		return PARAMETER_MISMATCH;

	try
	{
		//Scale image size that fit input size in model.
		Mat img1(pImage1->uiHeight, pImage1->uiWidth, CV_8UC3, (BYTE*)pImage1->pData);
		_image_preprocess(&img1);		

		//Do infernece and calculate the time cost.
		DWORD dwStart = GetTickCount();
		m_InferRequest.Infer();

		Blob::Ptr output_blob = m_InferRequest.GetBlob(m_OutputName);
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
		_image_preprocess(&img2);		

		//Do infernece and calculate the time cost.
		dwStart = GetTickCount();
		m_InferRequest.Infer();		

		Blob::Ptr output_blob1 = m_InferRequest.GetBlob(m_OutputName);
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

int	COPVO::ConverPtrToObjectDatas(int type, INT_PTR pInput, int size, INT_PTR *pOutput) {

	int nResult = OK;

	if (pInput == NULL)
		return PARAMETER_MISMATCH;

	switch (type) {

		case OBJECT_DETECTION_GENERAL:
			{
				*pOutput = _convert_to_objects(pInput, size);
			}
			break;

		case SPHEREFACE:
			{

			}
			break;

		default:
			nResult = NOT_IMPLEMENTED;
			break;
	}

	return nResult;
}

int	COPVO::FreeObjectDatas(ObjectDatas pOutput) {

	if (pOutput.pObjects) {
		free((void*)pOutput.pObjects);
		pOutput.pObjects = NULL;
	}

	return OK;
}


void COPVO::_show_model_info() {

	char szMsg[MAX_PATH] = { 0 };

	sprintf_s(szMsg, "Device : %s", m_Device);
	OutputDebugStringA(szMsg);

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

INT_PTR COPVO::_convert_to_objects(INT_PTR pInput, int size) {

	if (pInput == NULL || m_OutputInfo == NULL)
		return NULL;

	const SizeVector outputDims = m_OutputInfo->getTensorDesc().getDims();
	const int objectSize = outputDims[3];
	const SizeVector inputDims = m_InputInfo->getTensorDesc().getDims();
	const int nHeight = inputDims[2];
	const int nWidth = inputDims[3];
	vector<ObjectData> objects;

	const float *output = (float*)pInput;
	for (int curProposal = 0; curProposal < size; curProposal++) {
		int image_id = static_cast<int>(output[curProposal * objectSize + 0]);
		if (image_id < 0) {
			break;
		}

		ObjectData obj = { 0 };
		obj.conf = output[curProposal * objectSize + 2];
		obj.label = static_cast<int>(output[curProposal * objectSize + 1]);
		obj.x_min = static_cast<int>(output[curProposal * objectSize + 3] * nWidth) * m_fXRatio;
		obj.y_min = static_cast<int>(output[curProposal * objectSize + 4] * nHeight) * m_fYRatio;
		obj.x_max = static_cast<int>(output[curProposal * objectSize + 5] * nWidth) * m_fXRatio;
		obj.y_max = static_cast<int>(output[curProposal * objectSize + 6] * nHeight) * m_fYRatio;
		objects.push_back(obj);
	}

	/*OD_Datas *pDatas = new OD_Datas();
	if (pDatas) {
		pDatas->nCount = objects.size();
		ObjectData *data = (ObjectData*)malloc(sizeof(ObjectData) * pDatas->nCount);
		if (data) {
			memset(data, 0, sizeof(ObjectData) * pDatas->nCount);

			for (int n = 0; n < pDatas->nCount ; n++)
			{
				data[n] = objects[n];
			}
		}
		pDatas->pObjects = (INT_PTR)data;
		objects.clear();
	}*/

	return OK;
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