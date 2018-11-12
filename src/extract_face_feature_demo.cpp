#include <stdio.h>
#include <math.h>
#include "cpu.h"
#include "net.h"

int get_face_feature(const char* imgpath, ncnn::Mat& fc);

//demo
int main(int argc, char** argv)
{
    ncnn::Mat fc1;
   
    const char* image1path = argv[1];
    get_face_feature(image1path, fc1);
   
    for(int i=0; i<fc1.w*fc1.h*fc1.c; i++)
    {
        fprintf(stderr, "image1_features_num%d: %f\n",i, *(fc1.row(0)+i)); 
    }
    
    return 0;
}


//demo_sim
/*
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
    ncnn::Mat fc1;
    
    
    if (argc >= 4 || argc < 2)
    {
       printf("error:expect one(extract features) or two(simply test of similarity between two images) images\n");
       return -1;
    }
    if (argc >= 2 )
    {
        const char* image1path = argv[1];
        get_face_feature(image1path, fc1);
   
        for(int i=0; i<fc1.w*fc1.h*fc1.c; i++)
        {
            fprintf(stderr, "image1_features_num%d: %f\n",i, *(fc1.row(0)+i)); 
        }
    }
    if (argc >= 3)
    {
        const char* image2path = argv[2];
        ncnn::Mat fc2;

        get_face_feature(image2path, fc2);
        
        for(int i=0; i<fc2.w*fc2.h*fc2.c; i++)
        {
            fprintf(stderr, "image2_features_num%d: %f\n",i, *(fc2.row(0)+i)); 
        }
        float similarity;
        similarity = dot(fc1,fc2)/(sqrt(dot(fc1,fc1))*sqrt(dot(fc2,fc2)));
        fprintf(stderr, "similarity: %f\n", similarity); 
    }
    
    return 0;
}
*/
