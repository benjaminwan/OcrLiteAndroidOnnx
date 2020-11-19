#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

#include "OcrStruct.h"
#include "onnx/onnxruntime_cxx_api.h"
#include <opencv/cv.hpp>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace cv;
using namespace std;
using namespace Ort;

class CrnnNet {
public:
    ~CrnnNet();

    bool initModel(AAssetManager *mgr, Env &ortEnv, SessionOptions &sessionOptions);

    vector<TextLine> getTextLines(vector<Mat> &partImg);

private:
    unique_ptr<Session> session;
    vector<const char *> inputNames;
    vector<const char *> outputNames;
    vector<string> keys;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstHeight = 32;
    const int crnnCols = 5531;

    TextLine scoreToTextLine(const float *srcData, int h, int w);

    TextLine getTextLine(cv::Mat &src);

};


#endif //__OCR_CRNNNET_H__
