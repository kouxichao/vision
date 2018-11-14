#  this is a VISION project for hisi3559a.

# 编译成静态库：

```
cd $VISION
mkdir build
cd build
cmake ..
make -j$(nproc)
$VISION/lib 目录下生成libvision.a静态库
注：编译默认只编译crnn相关函数，facerecognization相关函数可添加编译选项-DNOT_ONLY_CRNN编译（相关函数未标准化）。
```
#crnn

使用说明：

 提供三个接口函数：

```
// 说明：根据识别出的文本框对里面的文字进行识别
// 初始化，用以加载ncnn lstm文字识别模型
void DKBoxTextRecognizationInit();

// 运行ncnn lstm文字识别，输出识别字符串
char* DKBoxTextRecognizationProcess(const char* imgfilename, int iHeight, int iWidth, DKSBox box, DKSBoxTextRecognizationParam param)

// 释放ncnn文字识别结构资源
void DKBoxTextRecognizationEnd();
```

  结构体定义：

```
typedef struct
{
    //左上角开始顺时针点坐标
    int x1;
    int y1;
    int x2;
    int y2;
    int x3;
    int y3;
    int x4;
    int y4;
}DKSBox;

typedef struct
{
        //等待添加
   const char* lexicon;//此选项未实现

}DKSBoxTextRecognizationParam;
```

  使用示例:
```
  https://github.com/kouxichao/crnn
```
