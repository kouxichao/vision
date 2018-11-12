#include<stdio.h>
#include<string>
#include<sqlite3.h>
#include"net.h"
#include"dlib/image_io.h"
using namespace dlib;
int normalize(ncnn::Mat& fc1);
int get_face_feature(array2d<rgb_pixel>& rgb_img, ncnn::Mat& fc);
int insert_feature(sqlite3* facefeatures, char* name, float* fc);
int update_feature(sqlite3* facefeatures, char* name, float* fc, int dyUP);
int create_db(const char* image_dir);
int face_recognise(const char* imagepath, std::string& recog_name);
int image_preprocess(const char* imagepath, array2d<rgb_pixel>&  face_chips);
int main(int argc, char* argv[])
{
//face_feature.db will be created for test.
//    const char* image_dir = argv[3];
//    create_db(image_dir);  

/*
//insert facefeatures
    sqlite3* facefeatures;
    int rc = sqlite3_open("face_feature.db", &facefeatures);
    
    if(rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(facefeatures));
        exit(0);
    }
    else
    {
        fprintf(stdout, "Opened database successfully\n");
    }
    const char* imagepath = argv[1];  
    char* name = argv[2];
    ncnn::Mat fc;
    array2d<rgb_pixel> rgb_img;

    image_preprocess(imagepath, rgb_img);
    get_face_feature(rgb_img, fc);
    normalize(fc);
 
    insert_feature(facefeatures, name, (float*)fc.data);
  //  update_feature(facefeatures, name, (float*)fc.data, 1);
    sqlite3_close(facefeatures);    

*/
//  reconise face 
 
    const char*   imagepath = argv[1];
    std::string result;
    face_recognise(imagepath, result);
    printf("name:%s\n", result.data());
    
    return 0;
}
