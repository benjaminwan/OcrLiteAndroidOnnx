#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "OcrLite.h"
#include "RRLib.h"
#include <iosfwd>

using namespace RRLib;

std::vector<const char *> getInputNames(std::unique_ptr<Ort::Session> &session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputNodes = session->GetInputCount();
    std::vector<const char *> inputNodeNames(numInputNodes);
    std::vector<int64_t> inputNodeDims;

    //printf("Number of inputs = %zu\n", numInputNodes);

    for (int i = 0; i < numInputNodes; i++) {
        // print input node names
        char *inputName = session->GetInputName(i, allocator);
        //printf("Input %d : name=%s\n", i, inputName);
        inputNodeNames[i] = inputName;

        // print input node types
        Ort::TypeInfo typeInfo = session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        //printf("Input %d : type=%d\n", i, type);

        // print input shapes/dims
        inputNodeDims = tensorInfo.GetShape();
        //printf("Input %d : num_dims=%zu\n", i, inputNodeDims.size());
        /*for (int j = 0; j < inputNodeDims.size(); j++)
            printf("Input %d : dim %d=%jd\n", i, j, inputNodeDims[j]);*/
    }
    return inputNodeNames;
}

std::vector<const char *> getOutputNames(std::unique_ptr<Ort::Session> &session) {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputNodes = session->GetOutputCount();
    std::vector<const char *> outputNodeNames(numOutputNodes);
    std::vector<int64_t> outputNodeDims;

    //printf("Number of outputs = %zu\n", numOutputNodes);

    for (int i = 0; i < numOutputNodes; i++) {
        // print input node names
        char *outputName = session->GetOutputName(i, allocator);
        //printf("Input %d : name=%s\n", i, outputName);
        outputNodeNames[i] = outputName;

        // print input node types
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensorInfo = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        //printf("Output %d : type=%d\n", i, type);

        // print input shapes/dims
        outputNodeDims = tensorInfo.GetShape();
        //printf("Output %d : num_dims=%zu\n", i, outputNodeDims.size());
        /*for (int j = 0; j < outputNodeDims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, outputNodeDims[j]);*/
    }
    return outputNodeNames;
}

char *readKeysFromAssets(AAssetManager *mgr) {
    //LOGI("readKeysFromAssets start...");
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
        return NULL;
    }
    char *buffer;
    /*获取文件名并打开*/
    AAsset *asset = AAssetManager_open(mgr, "keys.txt", AASSET_MODE_UNKNOWN);
    if (asset == NULL) {
        LOGE(" %s", "asset==NULL");
        return NULL;
    }
    /*获取文件大小*/
    off_t bufferSize = AAsset_getLength(asset);
    //LOGI("file size : %d", bufferSize);
    buffer = (char *) malloc(bufferSize + 1);
    buffer[bufferSize] = 0;
    int numBytesRead = AAsset_read(asset, buffer, bufferSize);
    //LOGI("readKeysFromAssets: %d", numBytesRead);
    /*关闭文件*/
    AAsset_close(asset);
    //LOGI("readKeysFromAssets exit...");
    return buffer;
}

void *getModelDataFromAssets(AAssetManager *mgr, const char *modelName, int &size) {
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
        return NULL;
    }
    AAsset *asset = AAssetManager_open(mgr, modelName, AASSET_MODE_UNKNOWN);
    if (asset == NULL) {
        LOGE(" %s", "asset==NULL");
        return NULL;
    }
    off_t bufferSize = AAsset_getLength(asset);
    void *modelData = malloc(bufferSize + 1);
    size = AAsset_read(asset, modelData, bufferSize);
    AAsset_close(asset);
    LOGI("model=%s, numBytesRead=%d", modelName, size);
    return modelData;
}

template<typename T, typename... Ts>
std::unique_ptr<T> makeUnique(Ts &&... params) {
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

OcrLite::OcrLite(JNIEnv *env, jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == NULL) {
        LOGE(" %s", "AAssetManager==NULL");
    }

    //init models
    //session options
    //sessionOptions.SetIntraOpNumThreads(numThread);
    sessionOptions.SetInterOpNumThreads(numThread);
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    int dbModelDataLength = 0;
    void *dbModelData = getModelDataFromAssets(mgr, "dbnet.onnx", dbModelDataLength);
    dbNet = std::make_unique<Ort::Session>(ortEnv, dbModelData, dbModelDataLength, sessionOptions);
    dbNetInputNames = getInputNames(dbNet);
    dbNetOutputNames = getOutputNames(dbNet);

    int angleModelDataLength = 0;
    void *angleModelData = getModelDataFromAssets(mgr, "angle_net.onnx", angleModelDataLength);
    angleNet = makeUnique<Ort::Session>(ortEnv, angleModelData, angleModelDataLength,
                                        sessionOptions);
    angleNetInputNames = getInputNames(angleNet);
    angleNetOutputNames = getOutputNames(angleNet);

    int crnnModelDataLength = 0;
    void *crnnModelData = getModelDataFromAssets(mgr, "crnn_lite_lstm.onnx", crnnModelDataLength);
    crnnNet = makeUnique<Ort::Session>(ortEnv, crnnModelData, crnnModelDataLength, sessionOptions);
    crnnNetInputNames = getInputNames(crnnNet);
    crnnNetOutputNames = getOutputNames(crnnNet);

    //int lineCount = 0;
    char *buffer = readKeysFromAssets(mgr);
    if (buffer != NULL) {//有该文件
        std::istringstream inStr(buffer);
        std::string line;
        //LOGI(" txt file  found");
        while (getline(inStr, line)) { // line中不包括每行的换行符
            keys.emplace_back(line);
            //lineCount++;
        }
        //LOGI("lineCount = %d", lineCount);
        free(buffer);
    } else {// 没有该文件
        LOGE(" txt file not found");
    }
    //LOGI("初始化完成!");
}

OcrLite::~OcrLite() {
    dbNet->release();
    angleNet->release();
    crnnNet->release();
}

std::vector<TextBox>
OcrLite::getTextBoxes(cv::Mat &src, ScaleParam &s,
                      float boxScoreThresh, float boxThresh, float minArea) {
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValsDBNet,
                                                                  normValsDBNet);

    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(),
                                                             inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = dbNet->Run(Ort::RunOptions{nullptr}, dbNetInputNames.data(), &inputTensor,
                                   dbNetInputNames.size(),
                                   dbNetOutputNames.data(), dbNetOutputNames.size());

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, floatArray, srcResize.rows * srcResize.cols * sizeof(float));

    std::vector<TextBox> rsBoxes;
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        std::vector<cv::Point> minBox;
        float minEdgeSize, allEdgeSize;
        getMiniBoxes(contours[i], minBox, minEdgeSize, allEdgeSize);

        if (minEdgeSize < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);

        if (score < boxScoreThresh)
            continue;

        for (int j = 0; j < minBox.size(); ++j) {
            minBox[j].x = (minBox[j].x / s.scaleWidth);
            minBox[j].x = (std::min)((std::max)(minBox[j].x, 0), s.srcWidth);

            minBox[j].y = (minBox[j].y / s.scaleHeight);
            minBox[j].y = (std::min)((std::max)(minBox[j].y, 0), s.srcHeight);
        }

        rsBoxes.emplace_back(TextBox(minBox, score));
    }
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

Angle scoreToAngle(const float *srcData, int w) {
    int angleIndex = 0;
    float maxValue = -1000.0f;
    for (int i = 0; i < w; i++) {
        if (i == 0)maxValue = srcData[i];
        else if (srcData[i] > maxValue) {
            angleIndex = i;
            maxValue = srcData[i];
        }
    }
    return Angle(angleIndex, maxValue);
}

Angle OcrLite::getAngle(cv::Mat &src) {
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(angleDstWidth, angleDstHeight));

    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValsAngle,
                                                                  normValsAngle);

    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(),
                                                             inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = angleNet->Run(Ort::RunOptions{nullptr}, angleNetInputNames.data(),
                                      &inputTensor,
                                      angleNetInputNames.size(),
                                      angleNetOutputNames.data(), angleNetOutputNames.size());

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    size_t count = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
    size_t rows = count / angleCols;
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    cv::Mat score(rows, angleCols, CV_32FC1);
    memcpy(score.data, floatArray, rows * angleCols * sizeof(float));

    return scoreToAngle((float *) score.data, angleCols);
}

TextLine OcrLite::scoreToTextLine(const float *srcData, int h, int w) {
    std::string strRes;
    int lastIndex = 0;
    std::vector<float> scores;
    for (int i = 0; i < h; i++) {
        //find max score
        int maxIndex = 0;
        float maxValue = -1000.f;
        for (int j = 0; j < w; j++) {
            if (srcData[i * w + j] > maxValue) {
                maxValue = srcData[i * w + j];
                maxIndex = j;
            }
        }
        if (maxIndex > 0 && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
    }
    return TextLine(strRes, scores);
}

TextLine OcrLite::getTextLine(cv::Mat &src) {
    float scale = (float) crnnDstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);

    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, crnnDstHeight));

    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValsCrnn,
                                                                  normValsCrnn);

    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(),
                                                             inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = crnnNet->Run(Ort::RunOptions{nullptr}, crnnNetInputNames.data(),
                                     &inputTensor,
                                     crnnNetInputNames.size(),
                                     crnnNetOutputNames.data(), crnnNetOutputNames.size());

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    size_t count = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
    size_t rows = count / crnnCols;
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    cv::Mat score(rows, crnnCols, CV_32FC1);
    memcpy(score.data, floatArray, rows * crnnCols * sizeof(float));

    return scoreToTextLine((float *) score.data, rows, crnnCols);
}

OcrResult OcrLite::detect(cv::Mat &src, cv::Mat &imgBox, ScaleParam &scale,
                          float boxScoreThresh, float boxThresh, float minArea,
                          float angleScaleWidth, float angleScaleHeight,
                          float textScaleWidth, float textScaleHeight) {

    int thickness = getThickness(src);

    LOGI("=====Start detect=====\n");
    LOGI("ScaleParam(sw:%d,sh:%d,dw:%d,dH%d,%f,%f)\n", scale.srcWidth, scale.srcHeight,
         scale.dstWidth, scale.dstHeight,
         scale.scaleWidth, scale.scaleHeight);

    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = getTextBoxes(src, scale, boxScoreThresh, boxThresh, minArea);
    LOGI("TextBoxesSize(%ld)\n", textBoxes.size());

    double endTextBoxesTime = getCurrentTime();
    double textBoxesTime = endTextBoxesTime - startTime;
    LOGI("getTextBoxesTime(%fms)\n", textBoxesTime);

    std::vector<Angle> angles;
    std::vector<TextLine> textLines;
    std::string strRes;
    for (int i = 0; i < textBoxes.size(); ++i) {
        LOGI("-----TextBox[%d] score(%f)-----\n", i, textBoxes[i].score);
        double startTextBox = getCurrentTime();
        cv::Mat angleImg;
        cv::RotatedRect rectAngle = getPartRect(textBoxes[i].box, angleScaleWidth,
                                                angleScaleHeight);
        RRLib::getRotRectImg(rectAngle, src, angleImg);
        LOGI("rectAngle(center.x=%f, center.y=%f, width=%f, height=%f, angle=%f)\n",
             rectAngle.center.x, rectAngle.center.y,
             rectAngle.size.width, rectAngle.size.height,
             rectAngle.angle);

        cv::Mat textImg;
        cv::RotatedRect rectText = getPartRect(textBoxes[i].box, textScaleWidth,
                                               textScaleHeight);
        RRLib::getRotRectImg(rectText, src, textImg);
        LOGI("rectText(center.x=%f, center.y=%f, width=%f, height=%f, angle=%f)\n",
             rectText.center.x, rectText.center.y,
             rectText.size.width, rectText.size.height,
             rectText.angle);

        //drawTextBox
        drawTextBox(imgBox, rectText, thickness);
        LOGI("TextBoxPos([x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d])\n",
             textBoxes[i].box[0].x, textBoxes[i].box[0].y,
             textBoxes[i].box[1].x, textBoxes[i].box[1].y,
             textBoxes[i].box[2].x, textBoxes[i].box[2].y,
             textBoxes[i].box[3].x, textBoxes[i].box[3].y);

        //Rotate Img
        if (angleImg.rows > 1.5 * angleImg.cols) {
            angleImg = matRotateClockWise90(angleImg);
            textImg = matRotateClockWise90(textImg);
        }

        //getAngle
        double startAngle = getCurrentTime();
        Angle angle = getAngle(angleImg);
        double endAngle = getCurrentTime();
        angle.time = endAngle - startAngle;

        //Log Angle
        LOGI("angle(index=%d, score=%f)\n", angle.index, angle.score);
        LOGI("getAngleTime(%fms)\n", angle.time);
        angles.push_back(angle);

        //Rotate Img
        if (angle.index == 0 || angle.index == 2) {
            textImg = matRotateClockWise180(textImg);
        }

        //getTextLine
        double startTextLine = getCurrentTime();
        TextLine textLine = getTextLine(textImg);
        double endTextLine = getCurrentTime();
        textLine.time = endTextLine - startTextLine;

        //Log textLine
        LOGI("textLine(%s)\n", textLine.line.c_str());
        std::ostringstream txtScores;
        for (int s = 0; s < textLine.scores.size(); ++s) {
            if (s == 0) txtScores << textLine.scores[s];
            txtScores << " ," << textLine.scores[s];
        }
        LOGI("textScores{%s}\n", std::string(txtScores.str()).c_str());
        LOGI("getTextLineTime(%fms)\n", textLine.time);
        textLines.push_back(textLine);

        //Log TextBox[i]Time
        double endTextBox = getCurrentTime();
        double timeTextBox = endTextBox - startTextBox;
        LOGI("TextBox[%i]Time(%fms)\n", i, timeTextBox);

        strRes.append(textLine.line);
        strRes.append("\n");
    }
    double endTime = getCurrentTime();
    LOGI("=====End detect=====\n");
    LOGI("FullDetectTime(%fms)\n", endTime - startTime);

    return OcrResult(textBoxes, textBoxesTime, angles, textLines, strRes);
}
