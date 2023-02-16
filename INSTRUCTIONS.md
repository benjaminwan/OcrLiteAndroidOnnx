# 使用说明

0. 根目录 build.gradle 添加

```groovy
repositories {
    google()
    jcenter()
    maven { url "https://jitpack.io" }
}
```

1. 添加 dependencies

```groovy
dependencies {
    implementation 'com.github.benjaminwan:OcrLiteAndroidOnnx:1.8.0'
}
```

2. 实例化OcrEngine

```
OcrEngine ocrEngine = new OcrEngine(this.getApplicationContext());
```

3. 调用detect方法

```
OcrResult result = ocrEngine.detect(img, boxImg, reSize);
```