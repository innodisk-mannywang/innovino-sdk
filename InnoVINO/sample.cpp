#include "InnoVINO.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <dlfcn.h>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "inference_engine.hpp"

using namespace cv;
using namespace InferenceEngine;
using namespace std;

int main() {

    InferRequest	m_InferRequest;
	InputInfo::Ptr	m_InputInfo;
	DataPtr			m_OutputInfo;
	string			m_Device;
	string			m_InputName;
	string			m_OutputName;
	float			m_fXRatio;
	float			m_fYRatio;


    //Integration Steps of OpenVINO
    //1. Initialize Core
    Core ie;
    //2. Read Model IRe
    CNNNetwork cnnNetwork = ie.ReadNetwork("/home/openvino-dev/Workspace/Project/innovino-sdk/InnoVINO/model/face-detection-0102.xml");		
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
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, "CPU");		
    //5. Create InferRequest
    m_InferRequest = exeNetwork.CreateInferRequest();

    return 1;
}