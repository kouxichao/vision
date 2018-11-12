#include <stdio.h>
#include "dlib/image_io.h"
#include "net.h"
using namespace dlib;
int get_face_feature(array2d<rgb_pixel>& rgb_img, ncnn::Mat& fc)
{
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("mobilefacenet.param");
    mobilefacenet.load_model("mobilefacenet.bin");
    
//    dlib::array2d<dlib::rgb_pixel> rgb_img;
//    load_image(rgb_img, imgpath);
    
    int col = (int)rgb_img.nc();
    int row = (int)rgb_img.nr();

    ncnn::Mat img;
    img.create(col*3, row, 1, 1);
   
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            *((unsigned char*)(img.data)+3*i*col+3*j)   = rgb_img[i][j].blue;
            *((unsigned char*)(img.data)+3*i*col+3*j+1) = rgb_img[i][j].green;
            *((unsigned char*)(img.data)+3*i*col+3*j+2) = rgb_img[i][j].red;
        }
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)img.data, ncnn::Mat::PIXEL_BGR2RGB, col, row, 112, 112);

    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(false);
    ex.input("data", in);
    ex.extract("fc1", fc);
    
    return 0;
}
