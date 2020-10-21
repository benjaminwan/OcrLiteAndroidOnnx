package com.benjaminwan.ocrlibrary

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap

class OcrEngine(context: Context) {
    init {
        System.loadLibrary("OcrLite")
        val ret = init(context.assets)
        if (!ret) throw IllegalArgumentException()
    }

    var numThread: Int = 4
    var padding: Int = 0
    var boxScoreThresh: Float = 0.6f
    var boxThresh: Float = 0.3f
    var miniArea: Float = 3f
    var angleScaleWidth: Float = 1.3f
    var angleScaleHeight: Float = 1.3f
    var textScaleWidth: Float = 1.6f
    var textScaleHeight: Float = 1.6f


    fun detect(input: Bitmap, output: Bitmap, reSize: Int) =
        detect(
            input, output, numThread, padding, reSize,
            boxScoreThresh, boxThresh, miniArea,
            angleScaleWidth, angleScaleHeight,
            textScaleWidth, textScaleHeight
        )

    external fun init(assetManager: AssetManager): Boolean
    external fun detect(
        input: Bitmap, output: Bitmap, numThread: Int, padding: Int, reSize: Int,
        boxScoreThresh: Float, boxThresh: Float, miniArea: Float,
        angleScaleWidth: Float, angleScaleHeight: Float,
        textScaleWidth: Float, textScaleHeight: Float
    ): String

}