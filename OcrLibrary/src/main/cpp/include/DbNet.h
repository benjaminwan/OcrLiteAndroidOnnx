#ifndef __OCR_DBNET_H__
#define __OCR_DBNET_H__

#include "OcrStruct.h"
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

class DbNet {
public:
    DbNet();

    ~DbNet();

    void setNumThread(int numOfThread);

    bool initModel(AAssetManager *mgr);

    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,
                                      float boxThresh, float unClipRatio);

private:
    Ort::Session *session;
    Ort::Env ortEnv = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;

    const float meanValues[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
    const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};
};


#endif //__OCR_DBNET_H__
