#include "AngleNet.h"
#include "OcrUtils.h"
#include <numeric>

AngleNet::~AngleNet() {
    session.release();
}

bool AngleNet::initModel(AAssetManager *mgr, Env &ortEnv, SessionOptions &sessionOptions) {
    int dbModelDataLength = 0;
    void *dbModelData = getModelDataFromAssets(mgr, "angle_net.onnx", dbModelDataLength);
    session = std::make_unique<Ort::Session>(ortEnv, dbModelData, dbModelDataLength,
                                             sessionOptions);

    inputNames = getInputNames(session);
    outputNames = getOutputNames(session);

    return true;
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

Angle AngleNet::getAngle(Mat &src) {

    vector<float> inputTensorValues = substractMeanNormalize(src, meanValues, normValues);

    array<int64_t, 4> inputShape{1, src.channels(), src.rows, src.cols};

    auto memoryInfo = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Value inputTensor = Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                   inputTensorValues.size(), inputShape.data(),
                                                   inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = session->Run(RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                     inputNames.size(),
                                     outputNames.data(), outputNames.size());

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    size_t count = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();
    size_t rows = count / angleCols;
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();

    Mat score(rows, angleCols, CV_32FC1);
    memcpy(score.data, floatArray, rows * angleCols * sizeof(float));

    return scoreToAngle((float *) score.data, angleCols);
}

vector<Angle> AngleNet::getAngles(vector<Mat> &partImgs,
                                  bool doAngle, bool mostAngle) {
    vector<Angle> angles;
    if (doAngle) {
        for (int i = 0; i < partImgs.size(); ++i) {
            //getAngle
            double startAngle = getCurrentTime();
            auto angleImg = adjustTargetImg(partImgs[i], dstWidth, dstHeight);
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;

            angles.emplace_back(angle);

        }
    } else {
        for (int i = 0; i < partImgs.size(); ++i) {
            Angle angle(-1, 0.f);
            angles.emplace_back(angle);
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        LOGI("Set All Angle to mostAngleIndex(%d)", mostAngleIndex);
        for (int i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}