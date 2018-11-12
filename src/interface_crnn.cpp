#include <stdio.h>
#include <omp.h>
#include "net.h"
#include "text_recognization.h"

static ncnn::Net netCrnn;

void DKBoxTextRecognizationInit()
{
    netCrnn.load_param("crnn.param");
    netCrnn.load_model("crnn.bin");
}

char* DKBoxTextRecognizationProcess(const char* imgfilename, int iHeight, int iWidth, DKSBox box, DKSBoxTextRecognizationParam param)
{
    //裁剪文字区域
    //dlib::array2d<dlib::rgb_pixel> rgb_img;
    //rgb_img.set_size(iHeight, iWidth);
    FILE *stream = NULL; 
    stream = fopen(imgfilename, "rb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    }

    unsigned char* yuvData = new unsigned char[iHeight*iHeight*2]
    fread(yuvData, 1, iHeight*iHeight*2, stream);
    fclose(stream);    
   
    int y_top = box.y1 > box.y2 ? box.y2 : box.y1;
    int y_bottom = box.y3 > box.y4 ? box.y3 : box.y4;
    int x_left = box.x1 > box.x4 ? box.x4 : box.x1;
    int x_right = box.x2 > box.x3 ? box.x2 : box.x3;
    y_top =  y_top > 0 ?  y_top : 0;
    x_left = x_left > 0 ? x_left : 0;
    y_bottom = y_bottom < iHeight ? y_bottom : iHeight; 
    x_right = x_right < iWidth ? x_right : iWidth; 
    int cols = x_right - x_left;
    int rows = y_bottom - y_top;
    
    ncnn::Mat img;
    img.create(iWidth, iHeight, 3, 1);
    int numOfPixel  = iHeight * iWidth; 
    int positionOfU = numOfPixel;
    int positionOfV = numOfPixel / 2 + numOfPixel;

    #pragma omp parallel for 
    for(int i = y_top; i < y_bottom; i++)
    {
        int startY = i * iWidth;
        int step = i * iWidth/2;
        int startU = positionOfU + step;
        int startV = positionOfV + step;
        for(int j=x_left; j< x_right; j++)
        {
	    int Y = startY + j;
	    int U = startU + j/2;
	    int V = startV + j/2;
            *((unsigned char*)(img.data)+3*i*cols+3*j)   = (unsigned char)(yuvData[Y] +  1.4075 * (yuvData[V] - 128));
            *((unsigned char*)(img.data)+3*i*cols+3*j+1) = (unsigned char)(yuvData[Y] - 0.3455 * (yuvData[U] - 128)  - 0.7169 * (yuvData[V] - 128));
            *((unsigned char*)(img.data)+3*i*cols+3*j+2) = (unsigned char)(yuvData[Y] + 1.779 * (yuvData[U] - 128));
        }
    }
    
    yuvData.free();

    //预处理并获取字符序列索引
    ncnn::Mat in,input_data;
    ncnn::Mat pred;
    in = ncnn::Mat::from_pixels((unsigned char*)img.data, ncnn::Mat::PIXEL_RGB, cols, rows);
    ncnn::resize_bilinear(in,input_data,100,32);
    input_data.reshape(100,32,1);
    
    #pragma omp parallel for  
    for(int i=0; i<100 * 32; i++)
    {
        *((float*)input_data.data+i) = ((*((float*)input_data.data+i))/255.f - 0.5)/0.5;
    }

    ncnn::Extractor ex = netCrnn.create_extractor();
    ex.set_light_mode(true);
    ex.input("data", input_data);
    ex.extract("preds", pred);

    //对输出字符索引解码得到字符串。
    float maxprob; 
    int pre_index = 0;
    char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz"; 
    static std::vector<char> result;

    for (int j=0; j<pred.h; j++)
    {
        int char_index = 0;
        maxprob = *((float*)pred.row(j));

        for (int i=0; i<pred.w; i++)
        {
            if (*((float*)pred.row(j)+i) > maxprob)
            {
                maxprob = *((float*)pred.row(j)+i);
                char_index = i;
            }
        }
        if(char_index != 0 && (j==0 || char_index != pre_index))
        { 
            result.push_back(alphabet[char_index-1]);            
        }      
        pre_index = char_index;
    }
    result.push_back('\0');    
    return result.data();
}

void DKBoxTextRecognizationEnd()
{
    netCrnn.clear();	
}
