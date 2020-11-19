#ifndef __OCR_UTILS_H__
#define __OCR_UTILS_H__

#include <opencv2/core.hpp>
#include "OcrStruct.h"
#include "onnx/onnxruntime_cxx_api.h"
#include <android/log.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

using namespace cv;
using namespace std;

#define TAG "OcrLite"
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,TAG,__VA_ARGS__)

template<typename T, typename... Ts>
static unique_ptr<T> makeUnique(Ts &&... params) {
    return unique_ptr<T>(new T(forward<Ts>(params)...));
}

template<typename T>
static double getMean(vector<T> &input) {
    auto sum = accumulate(input.begin(), input.end(), 0.0);
    return sum / input.size();
}

template<typename T>
static double getStdev(vector<T> &input, double mean) {
    if (input.size() <= 1) return 0;
    double accum = 0.0;
    for_each(input.begin(), input.end(), [&](const double d) {
        accum += (d - mean) * (d - mean);
    });
    double stdev = sqrt(accum / (input.size() - 1));
    return stdev;
}

double getCurrentTime();

ScaleParam getScaleParam(Mat &src, const float scale);

ScaleParam getScaleParam(Mat &src, const int targetSize);

RotatedRect getPartRect(vector<Point> &box, float scaleWidth, float scaleHeight);

int getThickness(Mat &boxImg);

void drawTextBox(Mat &boxImg, RotatedRect &rect, int thickness);

void drawTextBox(Mat &boxImg, const vector<Point> &box, int thickness);

void drawTextBoxes(Mat &boxImg, vector<TextBox> &textBoxes, int thickness);

Mat matRotateClockWise180(Mat src);

Mat matRotateClockWise90(Mat src);

Mat GetRotateCropImage(const Mat &src, vector<Point> box);

Mat adjustTargetImg(Mat &src, int dstWidth, int dstHeight);

int getMiniBoxes(vector<Point> &inVec,
                 vector<Point> &minBoxVec,
                 float &minEdgeSize, float &allEdgeSize
);

float boxScoreFast(Mat &mapmat, vector<Point> &_box);

void unClip(vector<Point> &minBoxVec, float allEdgeSize, vector<Point> &outVec, float unClipRatio);

vector<int> getAngleIndexes(vector<Angle> &angles);

vector<float> substractMeanNormalize(Mat &src, const float *meanVals, const float *normVals);

vector<const char *> getInputNames(unique_ptr<Ort::Session> &session);

vector<const char *> getOutputNames(unique_ptr<Ort::Session> &session);

void *getModelDataFromAssets(AAssetManager *mgr, const char *modelName, int &size);

#endif //__OCR_UTILS_H__
