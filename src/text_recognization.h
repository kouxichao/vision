#ifndef TEXT_RECOGNIZATION_H
#define TEXT_RECOGNIZATION_H

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
   const char* lexicon;
    
}DKSBoxTextRecognizationParam;

// 说明：根据识别出的文本框对里面的文字进行识别
// 初始化，用以加载ncnn lstm文字识别模型
void DKBoxTextRecognizationInit();

// 运行ncnn lstm文字识别，输出识别字符串
char* DKBoxTextRecognizationProcess(unsigned char* yuvData, int iHeight, int iWidth, DKSBox box, DKSBoxTextRecognizationParam param);

// 释放ncnn文字识别结构资源
void DKBoxTextRecognizationEnd();

#endif
