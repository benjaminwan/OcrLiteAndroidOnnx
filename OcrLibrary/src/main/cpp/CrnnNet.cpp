#include "CrnnNet.h"
#include "OcrUtils.h"
#include <fstream>
#include <numeric>

CrnnNet::~CrnnNet() {
    session.release();
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

bool CrnnNet::initModel(AAssetManager *mgr, Env &ortEnv, SessionOptions &sessionOptions) {
    int dbModelDataLength = 0;
    void *dbModelData = getModelDataFromAssets(mgr, "crnn_lite_lstm.onnx", dbModelDataLength);
    session = std::make_unique<Ort::Session>(ortEnv, dbModelData, dbModelDataLength,
                                             sessionOptions);

    inputNames = getInputNames(session);
    outputNames = getOutputNames(session);

    //load keys
    char *buffer = readKeysFromAssets(mgr);
    if (buffer != NULL) {
        istringstream inStr(buffer);
        string line;
        int size = 0;
        while (getline(inStr, line)) {
            keys.emplace_back(line);
            size++;
        }
        free(buffer);
        LOGI("keys size(%d)", size);
    } else {
        LOGE(" txt file not found");
        return false;
    }

    return true;
}

TextLine CrnnNet::scoreToTextLine(const float *srcData, int h, int w) {
    string strRes;
    int lastIndex = 0;
    int keySize = keys.size();
    vector<float> scores;
    for (int i = 0; i < h; i++) {
        int maxIndex = 0;
        float maxValue = -1000.f;
        //do softmax
        vector<float> exps(w);
        for (int j = 0; j < w; j++) {
            float expSingle = exp(srcData[i * w + j]);
            exps.at(j) = expSingle;
        }
        float partition = accumulate(exps.begin(), exps.end(), 0.0);//row sum
        for (int j = 0; j < w; j++) {
            float softmax = exps[j] / partition;
            if (softmax > maxValue) {
                maxValue = softmax;
                maxIndex = j;
            }
        }

        //no softmax
        /*for (int j = 0; j < w; j++) {
            if (srcData[i * w + j] > maxValue) {
                maxValue = srcData[i * w + j];
                maxIndex = j;
            }
        }*/
        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex - 1]);
        }
        lastIndex = maxIndex;
    }
    return TextLine(strRes, scores);
}

TextLine CrnnNet::getTextLine(Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);
    LOGI("=====getTextLine start 1====");
    Mat srcResize;
    resize(src, srcResize, Size(dstWidth, dstHeight));

    vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);

    array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    LOGI("=====getTextLine start 2====");

    Value inputTensor = Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                   inputTensorValues.size(), inputShape.data(),
                                                   inputShape.size());
    assert(inputTensor.IsTensor());

    LOGI("=====getTextLine start 3====");

    auto outputTensor = session->Run(RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                     inputNames.size(),
                                     outputNames.data(), outputNames.size());

    LOGI("=====getTextLine start 4====");

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    size_t count = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
    size_t rows = count / crnnCols;
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    Mat score(rows, crnnCols, CV_32FC1);
    memcpy(score.data, floatArray, rows * crnnCols * sizeof(float));

    return scoreToTextLine((float *) score.data, rows, crnnCols);
}

vector<TextLine> CrnnNet::getTextLines(vector<Mat> &partImg) {
    vector<TextLine> textLines;
    for (int i = 0; i < partImg.size(); ++i) {
        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;

        //Log textLine
        //Logger("textLine[%d](%s)\n", i, textLine.text.c_str());
        textLines.emplace_back(textLine);
        ostringstream txtScores;
        for (int s = 0; s < textLine.charScores.size(); ++s) {
            if (s == 0) {
                txtScores << textLine.charScores[s];
            } else {
                txtScores << " ," << textLine.charScores[s];
            }
        }
        //Logger("textScores[%d]{%s}\n", i, string(txtScores.str()).c_str());
        //Logger("crnnTime[%d](%fms)\n", i, textLine.time);
    }
    return textLines;
}