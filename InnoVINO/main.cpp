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

using namespace cv;

int main() {

    try
    {
        char *error = NULL;

        //get current path.
        char szPath[PATH_MAX];
        if (getcwd(szPath, sizeof(szPath)) != NULL) {
            printf("Current working dir: %s\n", szPath);
        } else {
            perror("getcwd() error");
            return 1;
        }

        //load libInnoVINO.so library
        char szLibraryPath[PATH_MAX] = {0};
        sprintf(szLibraryPath, "%s/%s", szPath, "libInnoVINO.so");

        printf("dlopen...\n");
        void* handle = dlopen(szLibraryPath, RTLD_LAZY);
        printf("dlopen...done\n");
        if(!handle)
        {
            printf("load library failed.");
            return 0;
        }

        IVINO_INIT IVINO_Init = NULL;
        IVINO_INFERENCE IVINO_Inference = NULL;
        IVINO_FREEOBJECTDATAS IVINO_FreeObjectDatas = NULL;
        IVINO_UNINIT IVINO_Uninit = NULL;

        printf("load function...\n");
        IVINO_Init = (IVINO_INIT)dlsym(handle, "IVINO_Init");
        IVINO_Inference = (IVINO_INFERENCE)dlsym(handle, "IVINO_Inference");
        IVINO_FreeObjectDatas = (IVINO_FREEOBJECTDATAS)dlsym(handle, "IVINO_FreeObjectDatas");
        IVINO_Uninit = (IVINO_UNINIT)dlsym(handle, "IVINO_Uninit");
        printf("load function...done\n");

        if((error = dlerror()) != NULL)
        {
            printf("load function failed, Error : %s\n", error);
            return 2;
        }

        //initial InnoVINO library
        char szBIN[PATH_MAX] = {0};
        sprintf(szBIN, "%s/model/%s.bin", szPath, "face-detection-0102");
        
        char szXML[PATH_MAX] = {0};
        sprintf(szXML, "%s/model/%s.xml", szPath, "face-detection-0102");

        unsigned long ulServiceId = 0;
        OMZ_Model model;
        model.lpBIN = szBIN;
        model.lpXML = szXML;
        if(IVINO_Init(&ulServiceId, &model) != 0){
            printf("IVINO_Init failed.");
            return -1;
        }

        //load image
        char szImage[PATH_MAX] = {0};
        sprintf(szImage, "%s/data/face_detection.jpg", szPath);
        printf(szImage);
        printf("\n");
        
        cv::Mat mat = imread(szImage);    

        ImageData img = {0};
        img.uiHeight = mat.rows;
        img.uiWidth = mat.cols;
        img.uiSize = img.uiHeight * img.uiWidth * mat.channels();
        img.pData = mat.data;

        ObjectDatas *objs = NULL;
        objs = IVINO_Inference(ulServiceId, &img, false);
        if(objs){
            for(int i = 0 ; i < objs->nCount ; i++){
                ObjectData* obj = (ObjectData*)(objs->pObjects + sizeof(ObjectData) * i);
                if(obj->conf < 0.9f || obj->label != 1)
                    continue;

                cv::rectangle(mat,cv::Point(obj->x_min, obj->y_min),cv::Point(obj->x_max, obj->y_max),Scalar(0,0,255), 2, 1, 0);
            }
        }

        imshow("face", mat);
        cv::waitKey(0);

        IVINO_FreeObjectDatas(ulServiceId, objs);

        IVINO_Uninit(ulServiceId);

        printf("dlclose...\n");
        if(handle){
            dlclose(handle);
        }
        printf("dlclose...done\n");
    }
    catch(const std::exception& e)
    {
        printf("exception : ");
        printf(e.what());
    }
    return 0;
}