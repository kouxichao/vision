// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include "dlib/image_io.h"
#include <math.h>
#include "net.h"

using namespace dlib;

static int detect_mobilefacenet(const array2d<rgb_pixel>& bgr, ncnn::Mat& prop)
{
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("mobilefacenet.param");
    mobilefacenet.load_model("mobilefacenet.bin");
    
    int col = (int)bgr.nc();
    int row = (int)bgr.nr();
    fprintf(stderr, "col:%d, row:%d \n", col, row);
    ncnn::Mat img;
    img.create(col, row, 3, 1);

    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
        {
            *((unsigned char*)(img.data)+3*i*col+3*j) = bgr[i][j].blue;
            *((unsigned char*)(img.data)+3*i*col+3*j+1) = bgr[i][j].green;
            *((unsigned char*)(img.data)+3*i*col+3*j+2) = bgr[i][j].red;
        }
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)img.data, ncnn::Mat::PIXEL_BGR, col, row, 112, 112);

    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ex.extract("fc1", prop);

    return 0;
}

float dot(ncnn::Mat& fc1, ncnn::Mat& fc2)
{
    float sum = 0;
    for(int i=0; i<fc1.w*fc1.h*fc1.c; i++)
    {
        sum += (*(fc1.row(0)+i)) * (*(fc2.row(0)+i));
    }
    return sum;
}

int main(int argc, char** argv)
{
    const char* image1path = argv[1];
    const char* image2path = argv[2];

    array2d<rgb_pixel> m1 ;
    array2d<rgb_pixel> m2 ;
    
    load_image(m1, image1path);
    load_image(m2, image2path);

    ncnn::Mat prop1,prop2;
    detect_mobilefacenet(m1, prop1);
    fprintf(stderr,"out_shape(%d):%d %d %d\n",prop1.dims, prop1.w, prop1.h, prop1.c);
    
    detect_mobilefacenet(m2, prop2);
    for(int i=0; i<prop1.w*prop1.h*prop1.c; i++)
    {
        fprintf(stderr, "num%d: %f\n",i, *(prop1.row(0)+i)); 
    }

    float similarity;
    similarity = dot(prop1,prop2)/(sqrt(dot(prop1,prop1))*sqrt(dot(prop2,prop2)));
    fprintf(stderr, "similarity: %f\n", similarity); 
    return 0;
}

